from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import RetrievalHitRate, RetrievalMRR


@dataclass
class PairwiseCanRankV3Args:
    base_feat_dim: int  # dimension of the heuristic features (e.g., string similarity, pagerank)
    embed_dim: int  # dimension of the sentence embedding
    pgrnk_stat_file: Path | str  # path to the pagerank statistics (e.g., pagerank_en.pkl)
    pgrnk_norm: Literal[
        "normalization", "none", "standardization"
    ] = "normalization"  # normalization method, normalization seems to be the best
    training_topk: int = 20  # top-k candidates to use for training
    training_add_missing_gold: bool = (
        True  # whether to add the gold candidate to the training set
    )


class PairwiseCanRankV3(BaseTorchCanRankModel[PairwiseCanRankV3Args]):
    VERSION = 100
    EVAL_BATCH_SIZE = 10000
    EXPECTED_EVAL_ARGS = {"features", "entity_desc_embedding", "header_embedding"}
    EXPECTED_ARGS = EXPECTED_EVAL_ARGS.union({"label"})

    def __init__(
        self,
        base_feat_dim: int,
        embed_dim: int,
        log_pagerank: Optional[tuple[float, float]] = None,
    ):
        super().__init__()

        self.log_pagerank = log_pagerank
        hidden_dim = 3 * base_feat_dim
        self.embed_dim = embed_dim

        self.context_mat_diag = nn.Parameter(torch.ones(self.embed_dim))
        # two hidden layers
        self.cmp = nn.Sequential(
            nn.Linear(base_feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.class_weights = nn.Parameter(
            torch.FloatTensor([1, 1]), requires_grad=False
        )
        self.args: Optional[PairwiseCanRankV3Args] = None
        self.save_hyperparameters()

        self.metrics = {}
        for subset in ["train", "val", "test"]:
            # note: skip the NIL or dangling entity as MRR & Hit@k aren't designed to measure NIL/dangling -- use precision/recall instead.
            setattr(self, f"{subset}_mrr", RetrievalMRR(empty_target_action="skip"))
            for k in [1, 3, 5, 10]:
                setattr(
                    self,
                    f"{subset}_hit@{k}",
                    RetrievalHitRate(empty_target_action="skip", k=k),
                )

    @classmethod
    def from_args(cls, args: PairwiseCanRankV3Args):
        pgrnk_stat = serde.pickle.deser(args.pgrnk_stat_file)
        if args.pgrnk_norm == "normalization":
            log_pagerank = (
                float(np.log(pgrnk_stat["min"] + 1e-20)),
                float(np.log(pgrnk_stat["max"]) - np.log(pgrnk_stat["min"] + 1e-20)),
            )
        elif args.pgrnk_norm == "standardization":
            pgrnk_mean = pgrnk_stat["meanlog"]
            pgrnk_std = pgrnk_stat["stdlog"]
            log_pagerank = (pgrnk_mean, pgrnk_std)
        else:
            assert args.pgrnk_norm != "none"
            log_pagerank = None

        model = cls(
            base_feat_dim=args.base_feat_dim,
            embed_dim=args.embed_dim,
            log_pagerank=log_pagerank,
        )
        model.args = args
        return model

    def forward(
        self, features, entity_desc_embedding, header_embedding, label=None
    ) -> BaseTorchCanRankOutput:
        if self.embed_dim == 0:
            ctx_score = torch.zeros((features.shape[0], 1)).to(features.device)
        else:
            ctx_score = (
                entity_desc_embedding
                * self.context_mat_diag.view(1, -1)
                * header_embedding
            ).sum(dim=1, keepdim=True)

        if self.log_pagerank is not None:
            norm_pg_rnk = (
                torch.log(features[:, -1:] + 1e-20) - self.log_pagerank[0]
            ) / self.log_pagerank[1]
            inputs = torch.cat([features[:, :-1], norm_pg_rnk, ctx_score], dim=1)
        else:
            inputs = torch.cat([features, ctx_score], dim=1)

        logits = self.cmp(inputs)

        if label is not None:
            loss = F.nll_loss(
                input=F.log_softmax(logits, dim=1),
                target=label,
                weight=self.class_weights,
            )
        else:
            loss = None

        return BaseTorchCanRankOutput(loss=loss, probs=F.softmax(logits, dim=1)[:, 1])

    def make_dataset(
        self,
        store: CanRankFeatStore,
        dsquery: str,
        verbose: bool = False,
    ):
        ds = store(
            CanRankFnArgs(
                dsquery=dsquery,
                text_embedding_model="sentence-transformers/all-mpnet-base-v2",
            ),
            [
                GetTopKCanBaseFeatFn,
                GetFreqMatchData,
                GetHeaderEmbeddingFn,
                GetEntityDescriptionEmbeddingFn,
            ],
        )
        ds.columns["features"] = np.concatenate(
            [
                ds.columns[store.get_func_name(GetTopKCanBaseFeatFn)],
                ds.columns[store.get_func_name(GetFreqMatchData)].reshape((-1, 1)),
            ],
            axis=1,
        ).astype(np.float32)
        ds.columns["header_embedding"] = ds.columns[
            store.get_func_name(GetHeaderEmbeddingFn)
        ]
        ds.columns["entity_desc_embedding"] = ds.columns[
            store.get_func_name(GetEntityDescriptionEmbeddingFn)
        ]
        return ds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        output = self._pl_step(batch, batch_idx, "train")
        return output.loss

    def validation_step(self, batch, batch_idx):
        self._pl_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._pl_step(batch, batch_idx, "test")

    def _pl_step(self, batch, batch_idx, prefix):
        batch_size = batch["label"].shape[0]
        output = self.forward(**{arg: batch[arg] for arg in self.EXPECTED_ARGS})

        self.log(
            f"{prefix}_loss",
            assert_not_null(output.loss),
            prog_bar=True,
            batch_size=batch_size,
        )
        if prefix != "train":
            # see usage here: https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
            preds, target, indexes = (
                output.probs,
                batch["label"],
                batch["cell_id"].long(),
            )
            # although the default of on_step and on_epoch for evaluation/test steps are correctly set by pytorch-lightning,
            # we specify them explicitly here for the training step. we need to do this as retrieval metrics are only intended to be used globally
            # see more: https://torchmetrics.readthedocs.io/en/v0.8.2/pages/retrieval.html and https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html
            getattr(self, f"{prefix}_mrr")(preds, target, indexes)
            getattr(self, f"{prefix}_hit@1")(preds, target, indexes)
            getattr(self, f"{prefix}_hit@3")(preds, target, indexes)
            getattr(self, f"{prefix}_hit@5")(preds, target, indexes)
            getattr(self, f"{prefix}_hit@10")(preds, target, indexes)
            self.log(
                f"{prefix}_mrr",
                getattr(self, f"{prefix}_mrr"),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )
            self.log(
                f"{prefix}_hit@1",
                getattr(self, f"{prefix}_hit@1"),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )
            self.log(
                f"{prefix}_hit@3",
                getattr(self, f"{prefix}_hit@3"),
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )
            self.log(
                f"{prefix}_hit@5",
                getattr(self, f"{prefix}_hit@5"),
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )
            self.log(
                f"{prefix}_hit@10",
                getattr(self, f"{prefix}_hit@10"),
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
            )
        return output


class PairwiseCanRankV3NoContext(PairwiseCanRankV3):
    def make_dataset(
        self,
        store: CanRankFeatStore,
        dsquery: str,
        verbose: bool = False,
    ):
        ds = store(
            CanRankFnArgs(
                dsquery=dsquery,
                text_embedding_model="sentence-transformers/all-mpnet-base-v2",
            ),
            [
                GetTopKCanBaseFeatFn,
                GetFreqMatchData,
                GetEmptyHeaderEmbeddingFn,
                GetEmptyEntityDescriptionEmbeddingFn,
            ],
        )
        ds.columns["features"] = np.concatenate(
            [
                ds.columns[store.get_func_name(GetTopKCanBaseFeatFn)],
                ds.columns[store.get_func_name(GetFreqMatchData)].reshape((-1, 1)),
            ],
            axis=1,
        ).astype(np.float32)
        ds.columns["header_embedding"] = ds.columns[
            store.get_func_name(GetEmptyHeaderEmbeddingFn)
        ]
        ds.columns["entity_desc_embedding"] = ds.columns[
            store.get_func_name(GetEmptyEntityDescriptionEmbeddingFn)
        ]
        return ds


class PairwiseCanRankV3SwitchContext(BaseTorchCanRankModel):
    EVAL_BATCH_SIZE = 1000
    EXPECTED_EVAL_ARGS = {
        "features",
        "entity_desc_embedding",
        "header_embedding",
        "has_header",
    }

    def __init__(self, context_model, no_context_model):
        super().__init__()
        self.context_model = context_model
        self.no_context_model = no_context_model

    @classmethod
    def from_args(cls, args: PairwiseCanRankV3Args):
        assert False

    @classmethod
    def load_from_checkpoint(cls, model_file: Path):
        context_model, no_context_model = str(model_file).split("::")
        context_model = PairwiseCanRankV3.load_from_checkpoint(Path(context_model))
        no_context_model = PairwiseCanRankV3.load_from_checkpoint(
            Path(no_context_model)
        )

        return cls(context_model, no_context_model)

    def forward(
        self, features, entity_desc_embedding, header_embedding, has_header, label=None
    ):
        assert label is None
        out1 = self.context_model(
            features=features[has_header],
            entity_desc_embedding=entity_desc_embedding[has_header],
            header_embedding=header_embedding[has_header],
            label=None,
        )
        out2 = self.no_context_model(
            features=features[~has_header],
            entity_desc_embedding=entity_desc_embedding[~has_header],
            header_embedding=header_embedding[~has_header],
            label=None,
        )

        probs = torch.zeros((features.shape[0],)).to(features.device) + 100
        probs[has_header] = out1.probs
        probs[~has_header] = out2.probs
        assert (probs != 100).all()

        return BaseTorchCanRankOutput(loss=None, probs=probs)

    def make_dataset(
        self,
        store: CanRankFeatStore,
        dsquery: str,
        verbose: bool = False,
    ):
        ds = store(
            CanRankFnArgs(
                dsquery=dsquery,
                text_embedding_model="sentence-transformers/all-mpnet-base-v2",
            ),
            [
                GetTopKCanBaseFeatFn,
                GetFreqMatchData,
                HasHeaderFn,
                GetHeaderEmbeddingFn,
                GetEntityDescriptionEmbeddingFn,
            ],
        )
        ds.columns["features"] = np.concatenate(
            [
                ds.columns[store.get_func_name(GetTopKCanBaseFeatFn)],
                ds.columns[store.get_func_name(GetFreqMatchData)].reshape((-1, 1)),
            ],
            axis=1,
        ).astype(np.float32)
        ds.columns["has_header"] = ds.columns[store.get_func_name(HasHeaderFn)]
        ds.columns["header_embedding"] = ds.columns[
            store.get_func_name(GetHeaderEmbeddingFn)
        ]
        ds.columns["entity_desc_embedding"] = ds.columns[
            store.get_func_name(GetEntityDescriptionEmbeddingFn)
        ]
        return ds
