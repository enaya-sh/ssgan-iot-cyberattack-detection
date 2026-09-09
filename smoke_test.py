import numpy as np
import torch

from src.improved_abc import ImprovedABC
from src.random_key import (
    CategoricalSpec,
    ContinuousSpec,
    IntegerSpec,
    RandomKeyDecoder,
)
from src.ssgan import SSGAN


def main():
    model = SSGAN(
        data_dim=10,
        num_classes=2,
        device=torch.device("cpu"),
    )

    labeled_x = torch.randn(43, 10)
    labeled_y = torch.randint(0, 2, (43,))
    unlabeled_x = torch.randn(35, 10)

    losses = model.train_step(
        labeled_x,
        labeled_y,
        unlabeled_x,
    )
    assert all(np.isfinite(v) for v in losses.values())

    schema = {
        "learning_rate": ContinuousSpec(0.0001, 0.01),
        "layers": IntegerSpec(1, 10),
        "activation": CategoricalSpec(
            ("relu", "linear", "tanh", "sigmoid")
        ),
    }
    decoder = RandomKeyDecoder(schema)
    decoded = decoder.decode(
        np.linspace(0.1, 0.9, decoder.dimension)
    )
    assert "learning_rate" in decoded

    optimizer = ImprovedABC(
        objective=lambda x: -float(
            np.sum((x - 0.3) ** 2)
        ),
        dimension=3,
        colony_size=6,
        max_iter=3,
        limit=2,
    )
    result = optimizer.optimize()
    assert np.isfinite(result.fitness)

    print("PASS: portfolio smoke test")


if __name__ == "__main__":
    main()
