from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Callable, Literal, Mapping, Optional, Union

import numpy as np
import xxx.postprocessing.reduce_numerical_noise as reduce_numerical_noise
from loguru import logger
from xxx.cangen import DatasetCandidateEntities, TableCandidateEntities
from xxx.dataset import Example, FullTable
from xxx.db import KGDB, KGDBArgs
from xxx.misc.ray_helper import ray_map, ray_put
from xxx.models.ont_class import OntologyClass
from xxx_retworkx import BaseEdge, BaseNode, RetworkXStrDiGraph
from xxx_retworkx import api as retworkx_api


@dataclass
class PredictColumnProbResult:
    col2type: dict[int, tuple[str, float]]
    col2type_freq: dict[int, list[tuple[str, float]]]


def predict_cta(
    kgdb: KGDB,
    examples: list[Example[FullTable]],
    cans: DatasetCandidateEntities,
    fn: Callable,
    params: dict,
    using_ray: bool = True,
) -> list[PredictColumnProbResult]:
    def exec(
        db: KGDB | KGDBArgs,
        ex: Example[FullTable],
        cans: TableCandidateEntities,
        fn: Callable,
        params: dict,
    ):
        return fn(db, ex, cans, **params)

    using_ray = using_ray and len(examples) > 1
    dbref = ray_put(kgdb.args, using_ray=using_ray)
    fnref = ray_put(fn, using_ray=using_ray)
    paramsref = ray_put(params, using_ray=using_ray)

    return ray_map(
        exec,
        [
            (
                dbref,
                ex,
                cans.get_table_candidates(ex.table.table.table_id),
                fnref,
                paramsref,
            )
            for ex in examples
        ],
        verbose=True,
        desc="column type annotation",
        is_func_remote=False,
        using_ray=using_ray,
    )


def predict_column_prob(
    kgdb: KGDB | KGDBArgs,
    ex: Example[FullTable],
    cans: TableCandidateEntities,
    jump_threshold: float = 0.1,
    jump_mode: Literal["exact", "percent"] = "exact",
    norm_eps: Optional[float] = 1e-6,
    tiebreak_eps: float = 1e-5,
    update_distance: int = 2,
    max_extend_distance: int = 2,
):
    kgdb = KGDB.get_instance(kgdb)
    db = kgdb.pydb

    entity_types = db.entity_types.cache()
    classes = db.classes.cache()
    ontcount = db.ontcount.cache()

    nrows, ncols = ex.table.table.shape()
    coltypes = {}
    coltypefreqs = {}

    for ci in range(ncols):
        type2row = {}
        for ri in range(nrows):
            if len(ex.table.links[ri, ci]) == 0:
                continue

            # leverage the fact that multiple links have been merged into one
            cellcan = cans.get_cell_candidates(ri, ci)
            for i in range(len(cellcan)):
                can_prob = cellcan.score[i]
                assert can_prob > 0.0
                for clsid in entity_types[cellcan.id[i]]:
                    if clsid not in type2row:
                        type2row[clsid] = np.zeros(nrows, dtype=np.float32)
                    type2row[clsid][ri] = max(type2row[clsid][ri], can_prob)

        if len(type2row) == 0:
            continue

        type2level, extended_type2row = extend_types(
            type2row, classes, update_distance, max_extend_distance
        )

        type_freq = [(type, row.mean()) for type, row in type2row.items()]
        type_freq = class_tiebreak(
            type_freq,
            classes,
            ontcount,
            norm_eps=norm_eps,
            tiebreak_eps=tiebreak_eps,
        )
        # [(type, wdclasses[type].label, score) for type, score in type_freq[:20]]
        best_type, best_score = max(type_freq, key=itemgetter(1))

        for level in range(0, max_extend_distance):
            # we should use type2row, because at the base level
            # we consider the most specific type. if we use extended_typerow
            # and the base type contains one of the parent of the other type
            # the parent freq will include freq of the children. why this is not
            # good? it prevents us from choosing the most specific type
            # and the heuristic that select the parent type when
            # NOTE: because we set best_type first, we did give it a chance, here we use extended
            # so we give the parent type discovered locally some priority
            type_freq = [
                (type, row.mean())
                for type, row in extended_type2row.items()
                if type2level[type] == level
            ]
            type_freq = class_tiebreak(
                type_freq,
                classes,
                ontcount,
                norm_eps=norm_eps,
                tiebreak_eps=tiebreak_eps,
            )

            type, score = max(type_freq, key=itemgetter(1))
            if best_type is None or can_jump(
                best_score, score, jump_threshold, jump_mode
            ):
                best_type = type
                best_score = score

        coltypes[ci] = (best_type, best_score)
        coltypefreqs[ci] = type_freq
    return PredictColumnProbResult(col2type=coltypes, col2type_freq=coltypefreqs)


def class_tiebreak(
    probs: list[tuple[str, float]],
    wdclasses: Mapping[str, OntologyClass],
    wdontcount: Mapping[str, int],
    norm_eps: Optional[float] = 1e-6,
    tiebreak_eps: float = 1e-5,
):
    newprobs = sorted(probs, key=itemgetter(1), reverse=True)
    if norm_eps is not None:
        newprobs = reduce_numerical_noise.normalize_probs(
            newprobs, eps=norm_eps, threshold=0.0
        )

    reduce_numerical_noise.tiebreak(
        newprobs,
        get_id=lambda x: x,
        id2popularity=wdontcount,
        id2ent=wdclasses,
        eps=tiebreak_eps,
    )
    return newprobs


def extend_types(
    type2row: dict[str, np.ndarray],
    classes: Mapping[str, OntologyClass],
    update_distance: int = 2,
    max_extend_distance: int = 2,
):
    # we keep track of the type of the node and the distance to the root
    base_ids = set(type2row.keys())
    type2level = {id: 0 for id in type2row}

    g = create_type_graph(type2level, classes)
    extended_type2row = {}
    for uid in retworkx_api.topological_sort(g):
        row = type2row[uid]
        for inedge in g.in_edges(uid):
            if inedge.key <= update_distance:
                row = np.maximum(type2row[inedge.source], row)
        extended_type2row[uid] = row

    for curr_dis in range(1, max_extend_distance + 1):
        new_ids = set()
        for id, dis in list(type2level.items()):
            if curr_dis - dis == 1:
                for pid in classes[id].parents:
                    if pid not in type2level:
                        new_ids.add(pid)
                        type2level[pid] = curr_dis
                        g.add_node(BaseNode(pid))
                        g.add_edge(BaseEdge(-1, id, pid, 1))
                        for inedge in g.in_edges(id):
                            g.add_edge(BaseEdge(-1, inedge.source, pid, inedge.key + 1))

        # only perform update for new nodes, there is a reason for that.
        for uid in new_ids:
            # always recompute from the unextended, otherwise, we may risk of aggregating pass a distance
            # for example, A is aggregated from B and C, C is aggregated from D, distance from A -> D may be larger
            # than update_distance, but if we use extended_type2row, A will use information from D as it use C, and C
            # is already use D.
            extended_type2row[uid] = np.maximum.reduce(
                [
                    type2row[inedge.source]
                    for inedge in g.in_edges(uid)
                    if inedge.key <= update_distance and type2level[inedge.source] == 0
                ]
            )

    return type2level, extended_type2row


def create_type_graph(
    ids: Union[set[str], dict[str, Any]],
    wdclasses: Mapping[str, OntologyClass],
):
    """Create a graph contain an edge from child to parent"""
    g = RetworkXStrDiGraph(check_cycle=False, multigraph=False)
    for id in ids:
        g.add_node(BaseNode(id))

    types = [wdclasses[id] for id in ids]
    for t1 in types:
        for ancestor_t1, distance in t1.ancestors.items():
            if ancestor_t1 in ids:
                # using edge key to store the distance
                g.add_edge(
                    BaseEdge(
                        -1,
                        t1.id,
                        ancestor_t1,
                        distance,
                    )
                )
    if retworkx_api.has_cycle(g):
        for u in g.iter_nodes():
            cycle = retworkx_api.digraph_find_cycle(g, u.id)
            if len(cycle) > 0:
                logger.error(
                    "Cycle detected: {}", [(e.source, e.target) for e in cycle]
                )
                raise ValueError(
                    "Cycle detected: " + str([(e.source, e.target) for e in cycle])
                )
        raise ValueError("Cycle detected")
    return g


def add_parents(g: RetworkXStrDiGraph, classes: Mapping[str, OntologyClass]):
    # get the list of ids
    ids = {u.id for u in g.nodes()}

    # retrieve the new parents
    parent_ids = set()
    for id in ids:
        parent_ids.update((pid for pid in classes[id].parents if pid not in ids))

    for parent_id in parent_ids:
        g.add_node(BaseNode(parent_id))

    for id in ids:
        for parent_id in parent_ids.intersection(classes[id].ancestors.keys()):
            g.add_edge(BaseEdge(-1, id, parent_id, 0))

    for parent_id in parent_ids:
        for id in ids.intersection(classes[parent_id].ancestors.keys()):
            g.add_edge(BaseEdge(-1, parent_id, id, 0))


def can_jump(from_val, to_val, threshold, mode):
    if mode == "exact":
        return to_val >= from_val + threshold
    elif mode == "percent":
        return to_val >= from_val * (1 + threshold)


def update_type_freq(g: RetworkXStrDiGraph, type2row: dict[str, np.ndarray]):
    new_type2row = type2row.copy()
    for uid in retworkx_api.topological_sort(g):
        parents = g.predecessors(uid)
        if len(parents) == 0:
            assert uid in new_type2row
            continue
        if uid in new_type2row:
            best_row = new_type2row[uid]
        else:
            best_row = new_type2row[parents[0].id]
        for pu in parents:
            best_row = np.maximum(best_row, new_type2row[pu.id])
        new_type2row[uid] = best_row
    return new_type2row
