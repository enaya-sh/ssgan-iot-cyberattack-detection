import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(model, loader):
    model.discriminator.eval()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device)
            logits, _ = model.discriminator(x)

            real_logits = logits[:, :model.num_classes]
            probs = torch.softmax(real_logits, dim=1)
            pred = probs.argmax(dim=1)

            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

            if model.num_classes == 2:
                y_score.extend(probs[:, 1].cpu().numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary" if model.num_classes == 2 else "weighted"),
        "precision": precision_score(
            y_true,
            y_pred,
            average="binary" if model.num_classes == 2 else "weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="binary" if model.num_classes == 2 else "weighted",
            zero_division=0,
        ),
    }

    if model.num_classes == 2 and len(set(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_score)

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
        tp = np.sum((y_true_np == 1) & (y_pred_np == 1))
        tn = np.sum((y_true_np == 0) & (y_pred_np == 0))
        fp = np.sum((y_true_np == 0) & (y_pred_np == 1))
        fn = np.sum((y_true_np == 1) & (y_pred_np == 0))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        metrics["g_mean"] = float(np.sqrt(sensitivity * specificity))

    return metrics
