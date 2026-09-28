import torch
from torch_geometric.data import Data

from src.train import (
    build_training_components,
    train_gcn,
)


def test_gcn_training_runs():

    x = torch.randn(20, 8)

    edge_index = torch.tensor(
        [
            list(range(19)) + list(range(1, 20)),
            list(range(1, 20)) + list(range(19)),
        ],
        dtype=torch.long,
    )

    y = torch.tensor(
        [0, 1] * 10,
        dtype=torch.long,
    )

    train_mask = torch.zeros(
        20,
        dtype=torch.bool,
    )

    val_mask = torch.zeros(
        20,
        dtype=torch.bool,
    )

    test_mask = torch.zeros(
        20,
        dtype=torch.bool,
    )

    train_mask[:12] = True
    val_mask[12:16] = True
    test_mask[16:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model, criterion, optimizer = (
        build_training_components(data)
    )

    history = train_gcn(
        model,
        data,
        criterion,
        optimizer,
        epochs=2,
    )

    assert len(history) == 1
    assert model is not None
