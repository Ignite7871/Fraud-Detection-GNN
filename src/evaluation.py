import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate(
    model,
    data,
    split: str = "val",
):
    """Evaluate a node-classification model on a graph split."""

    model.eval()

    with torch.no_grad():
        out = model(
            data.x,
            data.edge_index,
        )

        if split == "val":
            mask = data.val_mask
        elif split == "test":
            mask = data.test_mask
        elif split == "train":
            mask = data.train_mask
        else:
            raise ValueError(
                f"Unknown split: {split}"
            )

        logits = out[mask]

        preds = (
            logits.argmax(dim=1)
            .cpu()
            .numpy()
        )

        probs = (
            torch.softmax(
                logits,
                dim=1,
            )[:, 1]
            .cpu()
            .numpy()
        )

        labels = (
            data.y[mask]
            .cpu()
            .numpy()
        )

    accuracy = accuracy_score(
        labels,
        preds,
    )

    f1 = f1_score(
        labels,
        preds,
    )

    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(
            labels,
            probs,
        )

    return accuracy, f1, auc


def classification_details(
    model,
    data,
    split: str = "test",
):
    """Return confusion matrix and classification report."""

    model.eval()

    with torch.no_grad():
        out = model(
            data.x,
            data.edge_index,
        )

        if split == "test":
            mask = data.test_mask
        elif split == "val":
            mask = data.val_mask
        else:
            mask = data.train_mask

        logits = out[mask]

        y_true = (
            data.y[mask]
            .cpu()
            .numpy()
        )

        y_pred = (
            logits.argmax(dim=1)
            .cpu()
            .numpy()
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
    )

    return cm, report


def fraudulent_neighbor_ratio(
    data,
    test_idx,
) -> float:
    """Calculate average fraudulent-neighbor ratio."""

    edge_index_cpu = data.edge_index.cpu()
    fraud_mask = (
        data.y == 1
    ).cpu()

    neighbor_ratios = []

    for i in test_idx:

        neighbors = edge_index_cpu[1][
            edge_index_cpu[0] == i
        ]

        if neighbors.numel() > 0:

            ratio = (
                fraud_mask[neighbors]
                .float()
                .mean()
                .item()
            )

            neighbor_ratios.append(ratio)

    if not neighbor_ratios:
        return float("nan")

    return float(
        np.mean(neighbor_ratios)
    )
