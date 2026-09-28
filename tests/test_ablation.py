import torch
from torch_geometric.data import Data

from src.ablation import (
    add_neighbor_features,
    make_graph_only_features,
    mean_neighbor_features,
    replace_node_features,
)


def make_test_graph():
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )

    y = torch.tensor(
        [0, 1, 0],
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    data.train_mask = torch.tensor(
        [True, False, False]
    )

    data.val_mask = torch.tensor(
        [False, True, False]
    )

    data.test_mask = torch.tensor(
        [False, False, True]
    )

    return data


def test_graph_only_features():

    data = make_test_graph()

    graph_features = make_graph_only_features(
        data
    )

    assert graph_features.shape == (3, 1)

    assert torch.all(
        graph_features == 1.0
    )


def test_mean_neighbor_features():

    data = make_test_graph()

    neighbor_features = mean_neighbor_features(
        data.x,
        data.edge_index,
    )

    expected = torch.tensor(
        [
            [3.0, 4.0],
            [3.0, 4.0],
            [3.0, 4.0],
        ]
    )

    assert torch.allclose(
        neighbor_features,
        expected,
    )


def test_add_neighbor_features():

    data = make_test_graph()

    augmented = add_neighbor_features(
        data.x,
        data.edge_index,
    )

    assert augmented.shape == (3, 4)

    assert torch.allclose(
        augmented[:, :2],
        data.x,
    )


def test_replace_node_features_preserves_graph():

    data = make_test_graph()

    replacement = torch.ones(
        (3, 1)
    )

    new_data = replace_node_features(
        data,
        replacement,
    )

    assert torch.equal(
        new_data.edge_index,
        data.edge_index,
    )

    assert torch.equal(
        new_data.y,
        data.y,
    )

    assert torch.equal(
        new_data.train_mask,
        data.train_mask,
    )

    assert torch.equal(
        new_data.val_mask,
        data.val_mask,
    )

    assert torch.equal(
        new_data.test_mask,
        data.test_mask,
    )

    assert new_data.x.shape == (3, 1)
