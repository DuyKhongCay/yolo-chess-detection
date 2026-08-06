"""Export PyTorch YOLO model to ONNX format."""

from dataclasses import dataclass
from pathlib import Path
import draccus
from ultralytics import YOLO


@dataclass
class ExportOnnxConfig:
    """Configuration for exporting YOLO model to ONNX."""

    model: str = "weights/best.pt"
    imgsz: int = 640
    output_dir: str | None = None
    simplify: bool = True
    device: str = "cpu"


def export_onnx(cfg: ExportOnnxConfig) -> str:
    """Export YOLO model weights to ONNX format."""
    print(f"[+] Loading PyTorch model: {cfg.model}")
    model = YOLO(cfg.model)

    print(f"[+] Exporting model to ONNX (imgsz={cfg.imgsz}, device={cfg.device})...")
    export_path = model.export(
        format="onnx",
        imgsz=cfg.imgsz,
        simplify=cfg.simplify,
        device=cfg.device,
    )

    if cfg.output_dir:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = out_dir / Path(export_path).name
        if Path(export_path) != dest_path:
            import shutil
            shutil.move(export_path, dest_path)
            export_path = str(dest_path)

    print(f"[✓] ONNX Export completed: {export_path}")
    return str(export_path)


@draccus.wrap()
def main(cfg: ExportOnnxConfig):
    """Main entrypoint for ONNX export script."""
    export_onnx(cfg)


if __name__ == "__main__":
    main()
