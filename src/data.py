from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def load_elliptic_data(data_dir: str | Path):
    """
    Load the Elliptic Bitcoin transaction dataset.

    Returns:
        features_df: Raw transaction feature dataframe.
        labels_df: Transaction labels dataframe.
        edges_df: Transaction edge dataframe.
    """
    data_dir = Path(data_dir)

    features_path = data_dir / "elliptic_txs_features.csv"
    labels_path = data_dir / "elliptic_txs_classes.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    features_df = pd.read_csv(features_path, header=None)
    labels_df = pd.read_csv(labels_path)
    edges_df = pd.read_csv(edges_path)

    return features_df, labels_df, edges_df


def prepare_features(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    """
    Prepare node features and labels while preserving the notebook's
    original preprocessing behavior.
    """
    features_df = features_df.rename(
        columns={
            0: "txId",
            1: "time_step",
        }
    )

    tx_ids = features_df["txId"].values

    feature_cols = features_df.columns[2:]
    X = features_df[feature_cols].values

    labels_df = labels_df.rename(
        columns={
            "txId": "txId",
            "class": "label",
        }
    )

    labels_df = labels_df[labels_df["label"] != "unknown"].copy()
    labels_df["y"] = (labels_df["label"] == "1").astype(int)

    labeled = features_df.merge(
        labels_df[["txId", "y"]],
        on="txId",
        how="left",
    )

    y_full = labeled["y"].fillna(-1).astype(int).values
    time_steps = labeled["time_step"].values

    return X, y_full, tx_ids, time_steps


def build_graph(
    edges_df: pd.DataFrame,
    tx_ids: np.ndarray,
):
    """
    Convert transaction IDs into a bidirectional PyTorch Geometric
    edge index.
    """
    tx_id_to_idx = {
        tx_id: i
        for i, tx_id in enumerate(tx_ids)
    }

    edges_df = edges_df[
        edges_df["txId1"].isin(tx_id_to_idx)
        & edges_df["txId2"].isin(tx_id_to_idx)
    ]

    src = edges_df["txId1"].map(tx_id_to_idx).values
    dst = edges_df["txId2"].map(tx_id_to_idx).values

    edge_index = np.vstack((src, dst))

    # Preserve the original notebook behavior:
    # convert the directed edge list into an undirected graph.
    reverse_edges = np.vstack((dst, src))
    edge_index = np.concatenate(
        (edge_index, reverse_edges),
        axis=1,
    )

    return torch.tensor(
        edge_index,
        dtype=torch.long,
    )


def create_splits(
    y_full: np.ndarray,
    seed: int = 42,
):
    """
    Create the same stratified 64/16/20 train/validation/test split
    used in the original notebook.
    """
    labeled_indices = np.where(y_full != -1)[0]
    y_labeled = y_full[labeled_indices]

    train_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=0.2,
        stratify=y_labeled,
        random_state=seed,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        stratify=y_full[train_idx],
        random_state=seed,
    )

    return train_idx, val_idx, test_idx


def create_masks(
    num_nodes: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """Create PyTorch boolean masks for graph training and evaluation."""
    train_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    val_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    test_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def prepare_dataset(
    data_dir: str | Path = "data",
    seed: int = 42,
):
    """
    Run the complete dataset preparation pipeline.

    Returns:
        data: PyTorch Geometric Data object.
        X: Node feature matrix as NumPy array.
        y_full: Labels including -1 for unknown transactions.
        train_idx: Training indices.
        val_idx: Validation indices.
        test_idx: Test indices.
        time_steps: Original transaction time steps.
    """
    features_df, labels_df, edges_df = load_elliptic_data(
        data_dir
    )

    X, y_full, tx_ids, time_steps = prepare_features(
        features_df,
        labels_df,
    )

    edge_index = build_graph(
        edges_df,
        tx_ids,
    )

    x = torch.tensor(
        X,
        dtype=torch.float,
    )

    y = torch.tensor(
        y_full,
        dtype=torch.long,
    )

    train_idx, val_idx, test_idx = create_splits(
        y_full,
        seed=seed,
    )

    train_mask, val_mask, test_mask = create_masks(
        len(tx_ids),
        train_idx,
        val_idx,
        test_idx,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return (
        data,
        X,
        y_full,
        train_idx,
        val_idx,
        test_idx,
        time_steps,
    )


def create_temporal_splits(
    y_full: np.ndarray,
    time_steps: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """
    Create chronological train/validation/test splits using time steps.

    The split is performed at the time-step level so that a single
    time step cannot appear in multiple partitions.

    Returns:
        train_idx
        val_idx
        test_idx
        time_ranges: dictionary containing chronological boundaries.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be less than 1."
        )

    y_full = np.asarray(y_full)
    time_steps = np.asarray(time_steps)

    labeled_indices = np.where(y_full != -1)[0]

    if len(labeled_indices) == 0:
        raise ValueError("No labeled transactions were found.")

    labeled_times = time_steps[labeled_indices]

    unique_times = np.sort(
        np.unique(labeled_times)
    )

    if len(unique_times) < 3:
        raise ValueError(
            "At least three unique time steps are required."
        )

    train_end = int(
        np.floor(
            len(unique_times) * train_ratio
        )
    )

    val_end = int(
        np.floor(
            len(unique_times)
            * (train_ratio + val_ratio)
        )
    )

    # Guarantee at least one time step in every partition.
    train_end = max(1, train_end)

    val_end = max(
        train_end + 1,
        val_end,
    )

    val_end = min(
        val_end,
        len(unique_times) - 1,
    )

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    train_idx = labeled_indices[
        np.isin(
            labeled_times,
            train_times,
        )
    ]

    val_idx = labeled_indices[
        np.isin(
            labeled_times,
            val_times,
        )
    ]

    test_idx = labeled_indices[
        np.isin(
            labeled_times,
            test_times,
        )
    ]

    time_ranges = {
        "train_start": train_times[0],
        "train_end": train_times[-1],
        "val_start": val_times[0],
        "val_end": val_times[-1],
        "test_start": test_times[0],
        "test_end": test_times[-1],
    }

    return (
        train_idx,
        val_idx,
        test_idx,
        time_ranges,
    )


def restrict_graph_to_time(
    data: Data,
    time_steps: np.ndarray,
    max_time_step,
):
    """
    Create a graph view containing only edges whose endpoints occur
    at or before max_time_step.

    Node features remain available for evaluation, while future
    transaction relationships cannot influence the graph computation.
    """
    time_steps = np.asarray(time_steps)

    node_times = torch.as_tensor(
        time_steps,
        device=data.edge_index.device,
    )

    src_times = node_times[
        data.edge_index[0]
    ]

    dst_times = node_times[
        data.edge_index[1]
    ]

    keep_edges = (
        (src_times <= max_time_step)
        & (dst_times <= max_time_step)
    )

    temporal_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, keep_edges],
        y=data.y,
    )

    if hasattr(data, "train_mask"):
        temporal_data.train_mask = (
            data.train_mask.clone()
        )

    if hasattr(data, "val_mask"):
        temporal_data.val_mask = (
            data.val_mask.clone()
        )

    if hasattr(data, "test_mask"):
        temporal_data.test_mask = (
            data.test_mask.clone()
        )

    return temporal_data
