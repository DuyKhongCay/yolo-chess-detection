"""Export YOLO11n PyTorch model to Hailo-8 HEF format using Ultralytics and DFC API."""

import argparse
import os
from pathlib import Path

# Disable CUDA for TensorFlow backend in Hailo DFC to prevent cuDNN status 1002 errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    from hailo_sdk_client import ClientRunner
    DFC_AVAILABLE = True
except ImportError:
    DFC_AVAILABLE = False


def export_via_ultralytics(model_path: str, dataset_yaml: str, imgsz: int = 640, name: str = "hailo8"):
    """Export model to HEF using Ultralytics built-in Hailo exporter."""
    print(f"[+] Exporting via Ultralytics API for target '{name}'...")
    model = YOLO(model_path)
    export_path = model.export(
        format="hailo",
        name=name,
        imgsz=imgsz,
        data=dataset_yaml
    )
    print(f"[✓] Export completed: {export_path}")
    return export_path


def load_calibration_data(dataset_yaml: str, imgsz: int = 640, count: int = 64):
    """Load calibration images array for Hailo DFC optimization."""
    import yaml
    with open(dataset_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get('path', ''))
    val_sub = Path(data_cfg.get('val', 'images'))
    img_dir = base_path / val_sub if not val_sub.is_absolute() else val_sub

    img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
    img_paths = img_paths[:count]

    if not img_paths:
        raise FileNotFoundError(f"No images found in calibration directory: {img_dir}")

    calib_images = []
    for p in img_paths:
        img = Image.open(p).convert('RGB').resize((imgsz, imgsz))
        arr = np.array(img, dtype=np.float32) / 255.0
        calib_images.append(arr)

    return np.array(calib_images, dtype=np.float32)


def export_via_dfc(model_path: str, dataset_yaml: str, imgsz: int = 640, output_dir: str = "output"):
    """Export model to HEF step-by-step using Hailo DFC Python API."""
    if not DFC_AVAILABLE:
        raise RuntimeError("Hailo Dataflow Compiler (hailo_sdk_client) is not installed in current environment.")

    print("[+] Step 1: Exporting PyTorch model to ONNX...")
    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hef_path = out_dir / f"{Path(model_path).stem}_hailo8.hef"

    print("[+] Step 2: DFC Parsing ONNX model...")
    runner = ClientRunner(hw_arch='hailo8')
    runner.translate_onnx_model(onnx_path, model_name="yolo11n_chess")

    print("[+] Step 3: Preparing Calibration Data & Optimizing...")
    calib_data = load_calibration_data(dataset_yaml, imgsz=imgsz)
    runner.optimize(calib_data)

    print("[+] Step 4: Compiling Model to HEF...")
    runner.compile()
    hef_data = runner.get_hef()

    with open(hef_path, 'wb') as f:
        f.write(hef_data)

    print(f"[✓] DFC Export completed: {hef_path}")
    return str(hef_path)


def main():
    """Parse CLI arguments and launch conversion."""
    parser = argparse.ArgumentParser(description="Export YOLO model to Hailo HEF format.")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/duykhongcay/hailo_ws/chess_pieces_detection/runs/chess_detection_yolo11n/weights/best.pt",
        help="Path to PyTorch best.pt model file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/duykhongcay/lerobot_ws/chess_pieces_detection/datasets/dataset.yaml",
        help="Path to dataset.yaml file."
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--method", type=str, choices=["ultralytics", "dfc"], default="ultralytics", help="Conversion method.")
    parser.add_argument("--name", type=str, default="hailo8", help="Hailo hardware target name (e.g. hailo8, hailo8l).")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for DFC export.")

    args = parser.parse_args()

    if args.method == "ultralytics":
        export_via_ultralytics(args.model, args.dataset, imgsz=args.imgsz, name=args.name)
    else:
        export_via_dfc(args.model, args.dataset, imgsz=args.imgsz, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
