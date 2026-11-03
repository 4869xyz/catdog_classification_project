"""Classify one image with one final checkpoint."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
import torch

from catdog import CLASS_NAMES
from catdog.data import image_to_tensor
from catdog.utils import choose_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as image:
        tensor = image_to_tensor(image, checkpoint["image_size"]).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if checkpoint["model_name"] == "cnn":
            logits = (logits + model(torch.flip(tensor, dims=(-1,)))) / 2
        probabilities = logits.softmax(dim=1)[0].cpu().tolist()
    print(json.dumps({
        "image": str(args.image),
        "model": checkpoint["model_name"],
        "prediction": CLASS_NAMES[max(range(2), key=probabilities.__getitem__)],
        "probabilities": dict(zip(CLASS_NAMES, probabilities)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
