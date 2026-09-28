import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


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


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE network for node classification.

    Same shape/training interface as GCN (forward(x, edge_index) ->
    class logits), so it is a drop-in replacement for controlled
    architecture comparisons: only the graph convolution changes,
    everything else (hidden size, dropout, loss, optimizer, training
    loop) stays identical.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
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


class GAT(nn.Module):
    """Two-layer Graph Attention Network for node classification.

    Single attention head (heads=1) so the hidden dimension and
    parameter count stay directly comparable to GCN/GraphSAGE rather
    than growing with multi-head concatenation. Follows the standard
    GAT pattern (Velickovic et al., 2018): dropout on the input to
    each attention layer, ELU between layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=1,
            dropout=dropout,
        )

        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        x = nn.functional.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv1(x, edge_index)
        x = nn.functional.elu(x)

        x = nn.functional.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x
