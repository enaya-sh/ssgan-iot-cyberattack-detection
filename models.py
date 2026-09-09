import torch
import torch.nn as nn


def activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "linear": nn.Identity,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]()


class Generator(nn.Module):
    """
    MLP generator used for both noise-to-sample generation and
    discriminator-feature-to-sample reconstruction.
    """

    def __init__(
        self,
        noise_dim: int,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.36,
        activation: str = "relu",
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim

        def build_mlp(input_dim: int):
            layers = []
            current = input_dim
            for _ in range(max(num_layers - 1, 1)):
                layers += [
                    nn.Linear(current, hidden_dim),
                    activation_from_name(activation),
                    nn.Dropout(dropout),
                ]
                current = hidden_dim
            layers.append(nn.Linear(current, output_dim))
            return nn.Sequential(*layers)

        self.noise_path = build_mlp(noise_dim)
        self.reconstruction_path = build_mlp(feature_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.noise_path(z)

    def reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_path(features)


class Discriminator(nn.Module):
    """
    Semi-supervised discriminator with c + 1 outputs:
    c real classes plus one synthetic/fake class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.28,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = hidden_dim

        layers = []
        current = input_dim
        for _ in range(max(num_layers - 1, 1)):
            layers += [
                nn.Linear(current, hidden_dim),
                activation_from_name(activation),
                nn.Dropout(dropout),
            ]
            current = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features
