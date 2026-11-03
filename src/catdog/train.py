"""Train the final DNN, CNN, and RNN classifiers."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from catdog.data import CatDogDataset
from catdog.engine import run_epoch
from catdog.models import MODEL_NAMES, build_model
from catdog.utils import choose_device, save_checkpoint, set_seed

WEIGHT_DECAY = {"dnn": 1e-3, "cnn": 5e-4, "rnn": 1e-3}


def save_loss_curve(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o", markersize=3, label="Train loss")
    ax.plot(epochs, [row["val_loss"] for row in history], marker="o", markersize=3, label="Validation loss")
    lowest_loss = min(history, key=lambda row: row["val_loss"])
    best_accuracy = max(history, key=lambda row: (row["val_accuracy"], -row["val_loss"]))
    ax.axvline(lowest_loss["epoch"], color="green", linestyle="--",
               label=f"Min val loss: epoch {lowest_loss['epoch']}")
    ax.axvline(best_accuracy["epoch"], color="purple", linestyle=":",
               label=f"Best val accuracy: epoch {best_accuracy['epoch']}")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training history")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *MODEL_NAMES), default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")

    device = choose_device(args.device)
    for name in MODEL_NAMES if args.model == "all" else (args.model,):
        set_seed(args.seed)
        train_data = CatDogDataset(args.data_dir / "train", 64, augment=True)
        val_data = CatDogDataset(args.data_dir / "val", 64)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        model = build_model(name, 64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY[name])
        best_accuracy, best_loss, best_epoch = -1.0, math.inf, 0
        run_dir = args.output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "best.pt"
        history: list[dict] = []
        best_metrics: dict = {}
        summary = {
            "model": name,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "best_epoch": 0,
            "best_val": {},
            "lowest_val_loss": None,
            "lowest_val_loss_epoch": 0,
            "last_epoch": 0,
            "train_counts": train_data.class_counts,
            "val_counts": val_data.class_counts,
            "config": {
                "epochs": args.epochs, "seed": args.seed, "image_size": 64,
                "batch_size": 32, "learning_rate": 3e-4,
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "dropout": {
                    layer_name: layer.p for layer_name, layer in model.named_modules()
                    if isinstance(layer, torch.nn.Dropout)
                },
                "device": str(device),
            },
        }

        with (run_dir / "history.jsonl").open("w", encoding="utf-8") as history_file, \
                (run_dir / "training.log").open("w", encoding="utf-8") as log_file:
            for epoch in range(1, args.epochs + 1):
                train_metrics = run_epoch(model, train_loader, device, optimizer)
                val_metrics = run_epoch(model, val_loader, device)
                accuracy, loss = val_metrics["accuracy"], val_metrics["loss"]
                if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
                    best_accuracy, best_loss, best_epoch = accuracy, loss, epoch
                    best_metrics = val_metrics
                    save_checkpoint(
                        checkpoint_path, model, model_name=name, image_size=64,
                        epoch=epoch, validation_metrics=val_metrics,
                    )
                save_checkpoint(
                    run_dir / "last.pt", model, model_name=name, image_size=64,
                    epoch=epoch, validation_metrics=val_metrics,
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": loss,
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": accuracy,
                    "best_epoch": best_epoch,
                }
                history.append(row)
                history_file.write(json.dumps(row) + "\n")
                history_file.flush()
                if summary["lowest_val_loss"] is None or loss < summary["lowest_val_loss"]:
                    summary["lowest_val_loss"] = loss
                    summary["lowest_val_loss_epoch"] = epoch
                summary.update({"best_epoch": best_epoch, "best_val": best_metrics, "last_epoch": epoch})
                (run_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                line = (
                    f"{name} epoch {epoch:02d}/{args.epochs}: "
                    f"train_loss={row['train_loss']:.4f} val_loss={loss:.4f} "
                    f"train_acc={row['train_accuracy']:.4f} val_acc={accuracy:.4f} "
                    f"best_epoch={best_epoch} best_acc={best_accuracy:.4f}"
                )
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()
        save_loss_curve(history, run_dir / "loss_curve.png")
        print(f"{name}: best epoch {best_epoch}, val_acc={best_accuracy:.4f}; output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
