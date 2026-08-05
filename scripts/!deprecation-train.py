import os
from dataclasses import dataclass
from pathlib import Path
import draccus
from ultralytics import YOLO
from src.dataset import ChessREDTrainer

@dataclass
class TrainConfig:
    """Dataclass configuration for fine-tuning YOLOv11m on ChessRED using Draccus."""

    # Path to dataset configuration file
    data: str = "datasets/dataset.yaml"
    # Pretrained model checkpoint path or architecture file (.pt / .yaml)
    model: str = "yolo11m.pt"
    # Total number of training epochs
    epochs: int = 60
    # Batch size (-1 for AutoBatching based on available memory)
    batch: int = -1
    # Input image resolution in pixels
    imgsz: int = 640
    # CUDA device ID (e.g. '0', '0,1') or 'cpu'
    device: str = "cpu"
    # Experiment run name for saved outputs
    name: str = "chess_detection_yolo11m"
    # Early stopping patience (epochs without improvement)
    patience: int | None = 15
    # Optimizer algorithm (e.g. AdamW, SGD, Auto)
    optimizer: str | None = "AdamW"
    # Initial learning rate
    lr0: float = 0.001
    # Final learning rate fraction for Cosine decay
    lrf: float = 0.01
    # Enable Cosine Annealing learning rate schedule
    cos_lr: bool = True
    # Mosaic data augmentation probability
    mosaic: float = 1.0
    # Mixup data augmentation probability
    mixup: float = 0.1
    # Rotation data augmentation angle in degrees
    degrees: float = 15.0
    # Scale data augmentation gain factor
    scale: float = 0.5
    # Horizontal flip data augmentation probability
    fliplr: float = 0.5
    # Vertical flip data augmentation probability
    flipud: float = 0.0
    # Allow overwriting existing experiment directory
    exist_ok: bool = True
    verbose: bool = False
    val: bool = True
    resume: bool = False

@draccus.wrap()
def train(cfg: TrainConfig):
    """Main training execution loop wrapped with Draccus."""

    print("=== Starting YOLOv11m Fine-Tuning on ChessRED ===")
    print(f"Dataset config: {cfg.data}")
    print(f"Base model:     {cfg.model}")
    print(f"Resolution:     {cfg.imgsz}x{cfg.imgsz}")
    print(f"Epochs:         {cfg.epochs}, Batch Size: {cfg.batch}")

    # Load pretrained YOLOv11 model
    yolo_model = YOLO(cfg.model)

    # Train model using custom ChessREDTrainer to read COCO JSON directly
    results = yolo_model.train(
        data=str(cfg.data),
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        device=cfg.device,
        # project=cfg.project,
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
        exist_ok=cfg.exist_ok,
        verbose=cfg.verbose,
        save=True,
        plots=True,
        val=cfg.val,
        resume= cfg.resume
        
    )

    print(f"\n=== Training Completed Successfully ===")
    print(f"Checkpoints and metrics saved to: {cfg.name}")


if __name__ == "__main__":
    train()