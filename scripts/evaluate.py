"""Evaluation script for fine-tuned YOLOv11m model on ChessRED test set using Draccus."""

from dataclasses import dataclass, field
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import draccus
from ultralytics import YOLO
from scripts.dataset import ChessREDTrainer


@dataclass
class EvalConfig:
    """Dataclass configuration for model evaluation using Draccus."""
    model: str = field(
        default=str(PROJECT_ROOT / "models" / "yolo11m_chessred" / "weights" / "best.pt"),
        metadata={"help": "Path to trained model weights checkpoint (.pt)."},
    )
    data: str = field(
        default=str(PROJECT_ROOT / "datasets" / "dataset.yaml"),
        metadata={"help": "Path to dataset.yaml config file."},
    )
    imgsz: int = field(
        default=800,
        metadata={"help": "Input image resolution in pixels."},
    )
    split: str = field(
        default="test",
        metadata={"help": "Dataset split to evaluate on ('train', 'val', or 'test')."},
    )
    device: str = field(
        default="0",
        metadata={"help": "CUDA device ID or 'cpu'."},
    )


@draccus.wrap()
def evaluate(cfg: EvalConfig):
    """Run model evaluation on selected split wrapped with Draccus."""
    # Resolve absolute path for dataset config to avoid CWD issues
    data_path = Path(cfg.data)
    if not data_path.is_absolute():
        if (PROJECT_ROOT / data_path).exists():
            data_path = (PROJECT_ROOT / data_path).resolve()
        else:
            data_path = data_path.resolve()

    model_path = Path(cfg.model)
    if not model_path.exists():
        print(f"Error: Trained model weights file not found at {model_path}")
        print("Please ensure training has completed or specify correct --config_path or --model.")
        sys.exit(1)

    print(f"=== Evaluating Model on ChessRED Split: '{cfg.split}' ===")
    print(f"Model Checkpoint: {model_path}")
    print(f"Dataset Config:   {data_path}")
    print(f"Image Resolution: {cfg.imgsz}x{cfg.imgsz}")

    # Load fine-tuned YOLO model
    yolo_model = YOLO(str(model_path))

    # Run validation using custom ChessREDTrainer logic
    metrics = yolo_model.val(
        data=str(data_path),
        split=cfg.split,
        imgsz=cfg.imgsz,
        device=cfg.device,
        trainer=ChessREDTrainer,
        plots=True,
    )

    print("\n=== Evaluation Metrics ===")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    evaluate()
