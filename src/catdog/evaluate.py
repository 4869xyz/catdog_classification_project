"""Evaluate one final checkpoint on labeled images."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader

from catdog.data import CatDogDataset
from catdog.engine import run_epoch
from catdog.utils import choose_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(
        CatDogDataset(args.data_dir / "val", checkpoint["image_size"]),
        batch_size=32, shuffle=False,
    )
    flip_tta = checkpoint["model_name"] == "cnn"
    metrics = run_epoch(model, loader, device, flip_tta=flip_tta)
    print(json.dumps({
        "model": checkpoint["model_name"],
        "checkpoint_epoch": checkpoint["epoch"],
        "flip_tta": flip_tta,
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
