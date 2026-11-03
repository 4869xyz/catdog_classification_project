"""Model registry. Every model returns two raw class logits."""

from torch import nn

from catdog.models.cnn import CNNClassifier
from catdog.models.dnn import DNNClassifier
from catdog.models.rnn import RNNClassifier

MODEL_NAMES = ("dnn", "cnn", "rnn")


def build_model(name: str, image_size: int) -> nn.Module:
    if name == "dnn":
        return DNNClassifier(image_size)
    if name == "cnn":
        return CNNClassifier()
    if name == "rnn":
        return RNNClassifier(image_size)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
