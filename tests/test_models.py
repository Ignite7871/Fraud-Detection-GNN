import torch
from torch_geometric.data import Data

from src.models import GAT, GCN, GraphSAGE


def _make_test_graph():
    x = torch.randn(6, 10)

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    return Data(
        x=x,
        edge_index=edge_index,
    )


def test_gcn_forward_shape():

    data = _make_test_graph()

    model = GCN(
        in_channels=10,
        hidden_channels=8,
        out_channels=2,
    )

    output = model(
        data.x,
        data.edge_index,
    )

    assert output.shape == (6, 2)


def test_graphsage_forward_shape():

    data = _make_test_graph()

    model = GraphSAGE(
        in_channels=10,
        hidden_channels=8,
        out_channels=2,
    )

    output = model(
        data.x,
        data.edge_index,
    )

    assert output.shape == (6, 2)


def test_gat_forward_shape():

    data = _make_test_graph()

    model = GAT(
        in_channels=10,
        hidden_channels=8,
        out_channels=2,
    )

    output = model(
        data.x,
        data.edge_index,
    )

    assert output.shape == (6, 2)
