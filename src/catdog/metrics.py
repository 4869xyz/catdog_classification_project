"""Dataset-level loss and class-wise accuracy calculation."""

import torch
from torch import Tensor

from catdog import CLASS_NAMES


class ClassificationMetrics:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.total = 0
        # Rows are ground truth (cat, dog); columns are predictions (cat, dog).
        self.confusion = torch.zeros((2, 2), dtype=torch.long)

    def update(self, logits: Tensor, labels: Tensor, loss: Tensor) -> None:
        labels_cpu = labels.detach().to("cpu", dtype=torch.long)
        predicted = logits.detach().argmax(dim=1).to("cpu", dtype=torch.long)
        batch_size = labels_cpu.numel()
        self.loss_sum += loss.item() * batch_size
        self.total += batch_size
        self.confusion += torch.bincount(
            labels_cpu * 2 + predicted, minlength=4
        ).reshape(2, 2)

    def compute(self) -> dict:
        if self.total == 0:
            raise ValueError("Cannot calculate metrics for an empty dataset")
        correct = self.confusion.diag()
        support = self.confusion.sum(dim=1)
        return {
            "loss": self.loss_sum / self.total,
            "accuracy": correct.sum().item() / self.total,
            "per_class_accuracy": {
                name: correct[i].item() / support[i].item() if support[i].item() else None
                for i, name in enumerate(CLASS_NAMES)
            },
            "support": {name: support[i].item() for i, name in enumerate(CLASS_NAMES)},
            "confusion_matrix": self.confusion.tolist(),
        }
