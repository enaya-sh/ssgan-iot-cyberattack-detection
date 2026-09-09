import argparse
import random

import numpy as np
import torch

from .data import load_generic_csv, make_labeled_loader, make_unlabeled_loader
from .evaluation import evaluate_binary
from .ssgan import SSGAN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    set_seed(42)
    split = load_generic_csv(args.csv, args.label_column)

    # Thesis-reported values are used as documented defaults where this
    # compact portfolio runner can represent them directly.
    #
    # The thesis reports separate generator/discriminator batch sizes and
    # epoch counts. This reference runner does NOT claim exact reproduction
    # of that asynchronous training schedule.
    g_batch = 35
    d_batch = 43

    labeled_loader = make_labeled_loader(
        split.x_train, split.y_train, d_batch, True
    )
    unlabeled_loader = make_unlabeled_loader(
        split.x_unlabeled, g_batch, True
    )
    val_loader = make_labeled_loader(
        split.x_val, split.y_val, d_batch, False
    )
    test_loader = make_labeled_loader(
        split.x_test, split.y_test, d_batch, False
    )

    model = SSGAN(
        data_dim=split.x_train.shape[1],
        num_classes=split.num_classes,
        noise_dim=56,
        generator_layers=4,
        discriminator_layers=4,
        generator_dropout=0.36,
        discriminator_dropout=0.28,
        generator_lr=0.0024,
        discriminator_lr=0.0028,
        lambda_recon=0.35,
    )

    for epoch in range(args.epochs):
        unlabeled_iter = iter(unlabeled_loader)

        for labeled_x, labeled_y in labeled_loader:
            try:
                unlabeled_x = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_x = next(unlabeled_iter)

            model.train_step(labeled_x, labeled_y, unlabeled_x)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val = evaluate_binary(model, val_loader)
            print(f"Epoch {epoch + 1}: validation={val}")

    print("Test:", evaluate_binary(model, test_loader))


if __name__ == "__main__":
    main()
