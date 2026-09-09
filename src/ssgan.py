import torch
import torch.optim as optim

from .losses import (
    discriminator_unsupervised_loss,
    generator_unsupervised_loss,
    reconstruction_loss,
    supervised_loss,
)
from .models import Discriminator, Generator


class SSGAN:
    def __init__(
        self,
        data_dim: int,
        num_classes: int,
        noise_dim: int = 56,
        hidden_dim: int = 128,
        generator_layers: int = 4,
        discriminator_layers: int = 4,
        generator_dropout: float = 0.36,
        discriminator_dropout: float = 0.28,
        generator_lr: float = 0.0024,
        discriminator_lr: float = 0.0028,
        lambda_recon: float = 0.35,
        generator_activation: str = "relu",
        discriminator_activation: str = "relu",
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = num_classes
        self.lambda_recon = lambda_recon

        self.discriminator = Discriminator(
            input_dim=data_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=discriminator_layers,
            dropout=discriminator_dropout,
            activation=discriminator_activation,
        ).to(self.device)

        self.generator = Generator(
            noise_dim=noise_dim,
            feature_dim=self.discriminator.feature_dim,
            hidden_dim=hidden_dim,
            output_dim=data_dim,
            num_layers=generator_layers,
            dropout=generator_dropout,
            activation=generator_activation,
        ).to(self.device)

        self.g_opt = optim.Adam(self.generator.parameters(), lr=generator_lr)
        self.d_opt = optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )

    def train_step(self, labeled_x, labeled_y, unlabeled_x):
        labeled_x = labeled_x.to(self.device)
        labeled_y = labeled_y.to(self.device)
        unlabeled_x = unlabeled_x.to(self.device)

        # ---- Discriminator ----
        labeled_logits, _ = self.discriminator(labeled_x)
        sup = supervised_loss(
            labeled_logits, labeled_y, self.num_classes
        )

        unlabeled_logits, _ = self.discriminator(unlabeled_x)

        fake_batch = unlabeled_x.size(0)
        z = torch.randn(
            fake_batch, self.generator.noise_dim, device=self.device
        )
        fake_x = self.generator(z)
        fake_logits, _ = self.discriminator(fake_x.detach())

        unsup = discriminator_unsupervised_loss(
            unlabeled_logits, fake_logits, self.num_classes
        )
        d_loss = sup + unsup

        self.d_opt.zero_grad()
        d_loss.backward()
        self.d_opt.step()

        # ---- Generator ----
        z = torch.randn(
            fake_batch, self.generator.noise_dim, device=self.device
        )
        generated_x = self.generator(z)
        generated_logits, _ = self.discriminator(generated_x)
        g_adv = generator_unsupervised_loss(
            generated_logits, self.num_classes
        )

        # Thesis-aligned reconstruction path:
        # x -> discriminator feature map DF(x) -> generator reconstruction.
        _, real_features = self.discriminator(unlabeled_x)
        reconstructed_x = self.generator.reconstruct(
            real_features.detach()
        )
        recon = reconstruction_loss(unlabeled_x, reconstructed_x)

        g_loss = g_adv + self.lambda_recon * recon

        self.g_opt.zero_grad()
        g_loss.backward()
        self.g_opt.step()

        return {
            "d_loss": float(d_loss.detach().cpu()),
            "g_loss": float(g_loss.detach().cpu()),
            "supervised_loss": float(sup.detach().cpu()),
            "unsupervised_loss": float(unsup.detach().cpu()),
            "reconstruction_loss": float(recon.detach().cpu()),
        }
