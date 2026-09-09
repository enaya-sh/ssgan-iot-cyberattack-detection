import torch
import torch.nn.functional as F


EPS = 1e-8


def supervised_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    return F.cross_entropy(logits[:, :num_classes], labels)


def discriminator_unsupervised_loss(
    real_unlabeled_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    num_classes: int,
):
    real_probs = torch.softmax(real_unlabeled_logits, dim=1)
    fake_probs = torch.softmax(fake_logits, dim=1)

    p_fake_real = real_probs[:, num_classes]
    p_fake_fake = fake_probs[:, num_classes]

    real_term = -torch.mean(torch.log(1.0 - p_fake_real + EPS))
    fake_term = -torch.mean(torch.log(p_fake_fake + EPS))
    return real_term + fake_term


def generator_unsupervised_loss(fake_logits: torch.Tensor, num_classes: int):
    probs = torch.softmax(fake_logits, dim=1)
    p_fake = probs[:, num_classes]
    return -torch.mean(torch.log(1.0 - p_fake + EPS))


def reconstruction_loss(
    real_samples: torch.Tensor,
    reconstructed_samples: torch.Tensor,
):
    return F.mse_loss(reconstructed_samples, real_samples)
