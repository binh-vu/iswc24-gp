from dataclasses import dataclass
from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score, Precision, Recall


@dataclass
class BaseTorchOutput:
    loss: Optional[torch.Tensor]
    prob: torch.Tensor


class EdgeMLPV1(BaseTorchCPAModel):
    VERSION = 100
    EVAL_BATCH_SIZE = 1000
    EXPECTED_ARGS = {"label", "feature", "prop_embed", "header_embed"}
    EXPECTED_EVAL_ARGS = {"feature", "prop_embed", "header_embed"}

    def __init__(
        self,
        base_feat_dim: int,
        embed_dim: int,
    ):
        super().__init__()

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
        self.save_hyperparameters()

        self.metrics = {}
        self.precision = Precision("binary")
        self.recall = Recall("binary")
        self.f1 = F1Score("binary")

    def forward(self, feature, header_embed, prop_embed, label=None) -> BaseTorchOutput:
        if self.embed_dim == 0:
            ctx_score = torch.zeros((feature.shape[0], 1)).to(feature.device)
        else:
            ctx_score = (
                prop_embed * self.context_mat_diag.view(1, -1) * header_embed
            ).sum(dim=1, keepdim=True)

        inputs = torch.cat([feature, ctx_score], dim=1)
        logits = self.cmp(inputs)

        if label is not None:
            loss = F.nll_loss(
                input=F.log_softmax(logits, dim=1),
                target=label,
                weight=self.class_weights,
            )
        else:
            loss = None

        return BaseTorchOutput(loss=loss, prob=F.softmax(logits, dim=1)[:, 1])

    # def make_dataset(self, featstore)
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
        self.precision(output.prob, batch["label"])
        self.recall(output.prob, batch["label"])
        self.f1(output.prob, batch["label"])
        self.log(
            f"{prefix}_precision",
            self.precision,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}_recall",
            self.recall,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}_f1",
            self.f1,
            batch_size=batch_size,
        )
        return output
