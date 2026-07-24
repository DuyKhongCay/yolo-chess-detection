"""Fine-tuning script for YOLOv11m on ChessRED dataset using Draccus for configuration management."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import draccus
from ultralytics import YOLO
from scripts.dataset import ChessREDTrainer


@dataclass
class TrainConfig:
    """Dataclass configuration for fine-tuning YOLOv11m on ChessRED using Draccus."""
    data: str = field(
        default=str(PROJECT_ROOT / "datasets" / "dataset.yaml"),
        metadata={"help": "Path to dataset.yaml config file."},
    )
    model: str = field(
        default="yolo11m.pt",
        metadata={"help": "Pretrained model checkpoint path or architecture file (.pt / .yaml)."},
    )
    epochs: int = field(
        default=60,
        metadata={"help": "Number of training epochs."},
    )
    batch: int = field(
        default=16,
        metadata={"help": "Batch size for training."},
    )
    imgsz: int = field(
        default=640,
        metadata={"help": "Input image resolution in pixels."},
    )
    device: str = field(
        default="0",
        metadata={"help": "CUDA device ID or 'cpu'."},
    )
    project: str = field(
        default=str(PROJECT_ROOT / "models"),
        metadata={"help": "Directory to save training artifacts and model checkpoints."},
    )
    name: str = field(
        default="yolo11m_chessred",
        metadata={"help": "Experiment name for saved run."},
    )
    patience: int = field(
        default=15,
        metadata={"help": "Early stopping patience (epochs without improvement)."},
    )
    optimizer: str = field(
        default="AdamW",
        metadata={"help": "Optimizer algorithm (e.g. AdamW, SGD)."},
    )
    lr0: float = field(
        default=0.001,
        metadata={"help": "Initial learning rate."},
    )
    lrf: float = field(
        default=0.01,
        metadata={"help": "Final learning rate fraction for Cosine decay."},
    )
    cos_lr: bool = field(
        default=True,
        metadata={"help": "Enable Cosine Annealing learning rate schedule."},
    )
    mosaic: float = field(
        default=1.0,
        metadata={"help": "Mosaic data augmentation probability."},
    )
    mixup: float = field(
        default=0.1,
        metadata={"help": "Mixup data augmentation probability."},
    )
    degrees: float = field(
        default=15.0,
        metadata={"help": "Rotation augmentation degrees."},
    )
    scale: float = field(
        default=0.5,
        metadata={"help": "Scale augmentation factor."},
    )
    fliplr: float = field(
        default=0.5,
        metadata={"help": "Horizontal flip probability."},
    )
    flipud: float = field(
        default=0.0,
        metadata={"help": "Vertical flip probability (keep 0.0 for chess board)."},
    )


@draccus.wrap()
def train(cfg: TrainConfig):
    """Main training execution loop wrapped with Draccus."""
    # Resolve absolute path for dataset config to avoid CWD issues
    data_path = Path(cfg.data)
    if not data_path.is_absolute():
        if (PROJECT_ROOT / data_path).exists():
            data_path = (PROJECT_ROOT / data_path).resolve()
        else:
            data_path = data_path.resolve()

    print("=== Starting YOLOv11m Fine-Tuning on ChessRED ===")
    print(f"Dataset config: {data_path}")
    print(f"Base model:     {cfg.model}")
    print(f"Resolution:     {cfg.imgsz}x{cfg.imgsz}")
    print(f"Epochs:         {cfg.epochs}, Batch Size: {cfg.batch}")

    # Load pretrained YOLOv11 model
    yolo_model = YOLO(cfg.model)

    # Train model using custom ChessREDTrainer to read COCO JSON directly
    results = yolo_model.train(
        data=str(data_path),
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
        trainer=ChessREDTrainer,
        patience=cfg.patience,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        cos_lr=cfg.cos_lr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        degrees=cfg.degrees,
        scale=cfg.scale,
        fliplr=cfg.fliplr,
        flipud=cfg.flipud,
        save=True,
        plots=True,
        val=True,
    )

    print(f"\n=== Training Completed Successfully ===")
    print(f"Checkpoints and metrics saved to: {Path(cfg.project) / cfg.name}")


if __name__ == "__main__":
    train()
