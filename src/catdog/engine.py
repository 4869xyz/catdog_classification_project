"""One training or evaluation pass over a data loader."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from catdog.metrics import ClassificationMetrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    flip_tta: bool = False,
) -> dict:
    is_training = optimizer is not None
    if is_training and flip_tta:
        raise ValueError("Flip averaging is only used for evaluation")
    model.train(is_training)
    criterion = nn.CrossEntropyLoss()
    metrics = ClassificationMetrics()
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if flip_tta:
                logits = (logits + model(torch.flip(images, dims=(-1,)))) / 2
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            metrics.update(logits, labels, loss)
    return metrics.compute()
