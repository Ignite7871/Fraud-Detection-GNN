import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels,
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = nn.functional.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x
