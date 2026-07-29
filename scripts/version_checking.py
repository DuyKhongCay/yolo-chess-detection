import argparse
from pathlib import Path
from ultralytics import YOLO


def detect_yolo_version(weights_path: str) -> str:
    """Infer YOLO version and scale (e.g. yolo11n, yolov8s) using info() and model structure."""
    model = YOLO(weights_path)

    # 1. Try getting original model name from train_args if valid
    ckpt = getattr(model, "ckpt", {}) or {}
    train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    base_model = train_args.get("model", "")
    if base_model and Path(base_model).stem not in ("best", "last", "weights"):
        return Path(base_model).stem

    # 2. Try getting from model YAML config
    yaml_config = getattr(model, "yaml", {}) or {}
    yaml_file = yaml_config.get("yaml_file", "")
    if yaml_file:
        stem = Path(yaml_file).stem
        if stem and stem not in ("best", "last"):
            return stem

    # 3. Use model.info() params and layer module types to identify family & scale
    layers, params, gradients, flops = model.info(verbose=False)
    module_names = set(m.__class__.__name__ for m in model.model.modules())

    # Determine YOLO family from characteristic module layers
    if "C3k2" in module_names or "C2PSA" in module_names:
        family = "yolo11"
    elif "C2f" in module_names:
        family = "yolov8"
    elif "C3" in module_names:
        family = "yolov5"
    elif "RepNCSPELAN4" in module_names:
        family = "yolov9"
    elif "C2fCIB" in module_names:
        family = "yolov10"
    else:
        family = "yolov8"

    # Determine scale size from parameter count
    if params < 5_000_000:
        scale = "n"
    elif params < 15_000_000:
        scale = "s"
    elif params < 23_000_000 and family == "yolo11":
        scale = "m"
    elif params < 35_000_000:
        scale = "m" if family == "yolov8" else "l"
    elif params < 50_000_000:
        scale = "l"
    else:
        scale = "x"

    return f"{family}{scale}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify YOLO model version and scale.")
    parser.add_argument(
        "--weights",
        default="chess_pieces_detection/best.pt",
        help="Path to model weights file (.pt)",
    )

    args = parser.parse_args()
    model_type = detect_yolo_version(args.weights)
    print(model_type)
