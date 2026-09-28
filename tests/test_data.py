import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data import (
    build_graph,
    create_masks,
    create_splits,
    create_temporal_splits,
    restrict_graph_to_time,
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


def test_temporal_splits_are_chronological():

    y = np.array(
        [0, 1] * 10
    )

    time_steps = np.repeat(
        np.arange(1, 11),
        2,
    )

    train_idx, val_idx, test_idx, time_ranges = (
        create_temporal_splits(
            y,
            time_steps,
            train_ratio=0.6,
            val_ratio=0.2,
        )
    )

    train_times = set(
        time_steps[train_idx]
    )

    val_times = set(
        time_steps[val_idx]
    )

    test_times = set(
        time_steps[test_idx]
    )

    assert train_times
    assert val_times
    assert test_times

    assert max(train_times) < min(val_times)
    assert max(val_times) < min(test_times)

    assert not train_times & val_times
    assert not train_times & test_times
    assert not val_times & test_times

    combined = (
        set(train_idx)
        | set(val_idx)
        | set(test_idx)
    )

    assert combined == set(
        range(len(y))
    )

    assert time_ranges["train_end"] < time_ranges["val_start"]
    assert time_ranges["val_end"] < time_ranges["test_start"]


def test_restrict_graph_to_time_removes_future_edges():

    x = torch.zeros(
        (4, 2)
    )

    y = torch.tensor(
        [0, 1, 0, 1]
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    time_steps = np.array(
        [1, 1, 2, 3]
    )

    temporal_data = restrict_graph_to_time(
        data,
        time_steps,
        max_time_step=2,
    )

    edges = {
        tuple(edge)
        for edge in temporal_data.edge_index.t().tolist()
    }

    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (1, 2) in edges
    assert (2, 1) in edges

    assert (2, 3) not in edges
    assert (3, 2) not in edges
