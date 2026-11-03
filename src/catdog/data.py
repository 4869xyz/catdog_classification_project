"""Dataset discovery and image preprocessing shared by all three models."""

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

from catdog import CLASS_NAMES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_to_tensor(
    image: Image.Image, image_size: int, *, augment: bool = False,
) -> torch.Tensor:
    """Return a 3 x H x W float tensor scaled to [-1, 1]."""
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if augment and random.random() < 0.5:
        image = ImageOps.mirror(image)
    pixels = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).div_(127.5).sub_(1.0)


class CatDogDataset(Dataset):
    """Read images named cat.* or dog.* from one split directory.

    A nested layout (split/cat/*.jpg and split/dog/*.jpg) is also accepted.
    Invalid or ambiguous filenames fail early rather than silently changing labels.
    """

    def __init__(
        self, split_dir: str | Path, image_size: int, *, augment: bool = False,
    ):
        self.split_dir = Path(split_dir).expanduser().resolve()
        self.image_size = image_size
        self.augment = augment
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {self.split_dir}")

        samples: list[tuple[Path, int]] = []
        for path in sorted(self.split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = self._infer_label(path)
            samples.append((path, label))

        if not samples:
            raise ValueError(f"No supported images found in {self.split_dir}")
        counts = [sum(label == i for _, label in samples) for i in range(len(CLASS_NAMES))]
        if any(count == 0 for count in counts):
            raise ValueError(f"Both classes are required in {self.split_dir}; counts={counts}")
        self.samples = samples
        self.class_counts = dict(zip(CLASS_NAMES, counts))

    def _infer_label(self, path: Path) -> int:
        parent = path.parent.name.lower()
        if parent in CLASS_NAMES:
            return CLASS_NAMES.index(parent)
        prefix = path.name.lower().split(".", 1)[0]
        if prefix in CLASS_NAMES:
            return CLASS_NAMES.index(prefix)
        raise ValueError(f"Cannot infer cat/dog label from: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                tensor = image_to_tensor(image, self.image_size, augment=self.augment)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Cannot decode image: {path}") from exc
        return tensor, label
