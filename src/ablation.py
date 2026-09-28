import torch
from torch_geometric.data import Data


def replace_node_features(
    data: Data,
    node_features: torch.Tensor,
) -> Data:
    """
    Create a graph with the same labels, edges, and masks,
    but with a replacement node-feature matrix.
    """

    if node_features.size(0) != data.num_nodes:
        raise ValueError(
            "node_features must contain one row per node."
        )

    if node_features.device != data.x.device:
        node_features = node_features.to(
            data.x.device
        )

    new_data = Data(
        x=node_features,
        edge_index=data.edge_index,
        y=data.y,
    )

    for mask_name in (
        "train_mask",
        "val_mask",
        "test_mask",
    ):
        if hasattr(data, mask_name):
            setattr(
                new_data,
                mask_name,
                getattr(data, mask_name).clone(),
            )

    return new_data


def make_graph_only_features(
    data: Data,
) -> torch.Tensor:
    """
    Create constant node features for a graph-only GCN.

    Every node receives the same constant feature value.
    Therefore, the GCN's predictive information must come
    from graph connectivity and message passing rather than
    transaction-level features.
    """

    return torch.ones(
        (
            data.num_nodes,
            1,
        ),
        dtype=data.x.dtype,
        device=data.x.device,
    )


def mean_neighbor_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the mean feature vector of each node's one-hop neighbors.

    Nodes without neighbors receive a zero vector.

    This function does not use labels.
    """

    if edge_index.numel() == 0:
        return torch.zeros_like(x)

    src = edge_index[0]
    dst = edge_index[1]

    neighbor_sum = torch.zeros_like(x)

    neighbor_sum.index_add_(
        0,
        dst,
        x[src],
    )

    neighbor_count = torch.zeros(
        (
            x.size(0),
            1,
        ),
        dtype=x.dtype,
        device=x.device,
    )

    ones = torch.ones(
        (
            dst.size(0),
            1,
        ),
        dtype=x.dtype,
        device=x.device,
    )

    neighbor_count.index_add_(
        0,
        dst,
        ones,
    )

    return (
        neighbor_sum
        / neighbor_count.clamp_min(1.0)
    )


def add_neighbor_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate original node features with one-hop
    mean-neighbor features.
    """

    neighbor_features = mean_neighbor_features(
        x,
        edge_index,
    )

    return torch.cat(
        [
            x,
            neighbor_features,
        ],
        dim=1,
    )
