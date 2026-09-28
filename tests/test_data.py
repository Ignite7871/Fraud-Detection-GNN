import numpy as np
import pandas as pd
import torch

from src.data import (
    build_graph,
    create_masks,
    create_splits,
)


def test_build_graph_is_bidirectional():

    edges = pd.DataFrame(
        {
            "txId1": [1, 2],
            "txId2": [2, 3],
        }
    )

    tx_ids = np.array([1, 2, 3])

    edge_index = build_graph(
        edges,
        tx_ids,
    )

    assert edge_index.shape == (2, 4)

    edge_set = {
        tuple(edge)
        for edge in edge_index.t().tolist()
    }

    assert (0, 1) in edge_set
    assert (1, 0) in edge_set
    assert (1, 2) in edge_set
    assert (2, 1) in edge_set


def test_masks_are_disjoint():

    train_idx = np.array([0, 1])
    val_idx = np.array([2])
    test_idx = np.array([3, 4])

    train_mask, val_mask, test_mask = create_masks(
        5,
        train_idx,
        val_idx,
        test_idx,
    )

    assert not torch.any(
        train_mask & val_mask
    )

    assert not torch.any(
        train_mask & test_mask
    )

    assert not torch.any(
        val_mask & test_mask
    )

    assert train_mask.sum() == 2
    assert val_mask.sum() == 1
    assert test_mask.sum() == 2


def test_splits_are_disjoint():

    y = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
        * 10
    )

    train, val, test = create_splits(
        y,
        seed=42,
    )

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0

    combined = set(train) | set(val) | set(test)

    assert combined == set(range(len(y)))
