import torch
from torch_geometric.data import Data

from src.models import GCN


def test_gcn_forward_shape():

    x = torch.randn(6, 10)

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    model = GCN(
        in_channels=10,
        hidden_channels=8,
        out_channels=2,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
    )

    output = model(
        data.x,
        data.edge_index,
    )

    assert output.shape == (6, 2)
