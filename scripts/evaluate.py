"""Evaluation script for fine-tuned YOLOv11m model on ChessRED test set using Draccus."""

from dataclasses import dataclass, field
import draccus
from ultralytics import YOLO
from src.dataset import ChessREDValidator


@dataclass
class EvalConfig:
    """Dataclass configuration for model evaluation using Draccus."""
    model: str | None
    data: str | None
    imgsz: int | None
    split: str | None
    device: str | None


@draccus.wrap()
def evaluate(cfg: EvalConfig):
    """Run model evaluation on selected split wrapped with Draccus."""
    # # Resolve absolute path for dataset config to avoid CWD issues
    # data_path = Path(cfg.data)
    # if not data_path.is_absolute():
    #     if (PROJECT_ROOT / data_path).exists():
    #         data_path = (PROJECT_ROOT / data_path).resolve()
    #     else:
    #         data_path = data_path.resolve()

    # model_path = Path(cfg.model)
    # if not model_path.exists():
    #     print(f"Error: Trained model weights file not found at {model_path}")
    #     print("Please ensure training has completed or specify correct --config_path or --model.")
    #     sys.exit(1)

    print(f"=== Evaluating Model on ChessRED Split: '{cfg.split}' ===")
    print(f"Model Checkpoint: {cfg.model}")
    print(f"Dataset Config:   {cfg.data}")
    print(f"Image Resolution: {cfg.imgsz}x{cfg.imgsz}")

    # Load fine-tuned YOLO model
    yolo_model = YOLO(str(cfg.model))

    # Run validation using custom ChessREDValidator logic
    metrics = yolo_model.val(
        data=str(cfg.data),
        split=cfg.split,
        imgsz=cfg.imgsz,
        device=cfg.device,
        validator=ChessREDValidator,
        plots=True,
    )

    print("\n=== Evaluation Metrics ===")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    evaluate()
