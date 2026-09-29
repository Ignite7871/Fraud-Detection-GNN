import torch
from torch_geometric.data import Data

from src.evaluation import predict_probabilities
from src.models import GCN


def test_predict_probabilities_shapes_and_range():

    x = torch.randn(6, 10)

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )

    y = torch.tensor(
        [0, 1, 0, 1, 0, 1],
        dtype=torch.long,
    )

    test_mask = torch.tensor(
        [False, False, False, True, True, True]
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )
    data.test_mask = test_mask
    data.val_mask = test_mask
    data.train_mask = ~test_mask

    model = GCN(
        in_channels=10,
        hidden_channels=8,
        out_channels=2,
    )

    y_true, y_pred, y_prob = predict_probabilities(
        model,
        data,
        split="test",
    )

    assert y_true.shape == (3,)
    assert y_pred.shape == (3,)
    assert y_prob.shape == (3,)

    assert set(y_pred.tolist()) <= {0, 1}
    assert ((y_prob >= 0.0) & (y_prob <= 1.0)).all()


def test_predict_probabilities_unknown_split_raises():

    x = torch.randn(4, 5)
    edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.test_mask = torch.tensor([True, True, False, False])

    model = GCN(in_channels=5, hidden_channels=4, out_channels=2)

    try:
        predict_probabilities(model, data, split="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
