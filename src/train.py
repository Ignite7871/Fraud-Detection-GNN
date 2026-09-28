import torch
from torch import nn

from .evaluation import evaluate
from .models import GCN


def build_training_components(
    data,
    hidden_channels: int = 64,
    dropout: float = 0.5,
    learning_rate: float = 0.001,
    weight_decay: float = 5e-4,
):
    """Build the GCN, weighted loss, and optimizer."""

    train_labels = data.y[data.train_mask]

    class_counts = torch.bincount(
        train_labels,
        minlength=2,
    )

    class_weights = (
        train_labels.size(0)
        / (2.0 * class_counts)
    ).float().to(data.x.device)

    model = GCN(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=2,
        dropout=dropout,
    ).to(data.x.device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return model, criterion, optimizer


def train_gcn(
    model,
    data,
    criterion,
    optimizer,
    epochs: int = 100,
):
    """Train the GCN and record validation metrics."""

    val_history = []

    for epoch in range(1, epochs + 1):

        model.train()

        optimizer.zero_grad()

        out = model(
            data.x,
            data.edge_index,
        )

        loss = criterion(
            out[data.train_mask],
            data.y[data.train_mask],
        )

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:

            val_acc, val_f1, val_auc = evaluate(
                model,
                data,
                "val",
            )

            val_history.append(
                (
                    epoch,
                    val_acc,
                    val_f1,
                    val_auc,
                )
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Val Acc {val_acc:.4f} | "
                f"Val F1 {val_f1:.4f} | "
                f"Val AUC {val_auc:.4f}"
            )

    return val_history
