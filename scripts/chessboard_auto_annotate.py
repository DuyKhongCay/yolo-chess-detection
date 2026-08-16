"""Auto-annotate chessboard images and export to YOLO segmentation dataset format."""

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import cv2
import draccus
import numpy as np
import yaml

import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_segmentor import BoardSegmentor


@dataclass
class AutoAnnotateConfig:
    images_dir: str = ""
    output_dir: str = ""
    model_path: str = ""
    image_count: int = -1
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    class_id: int = 0
    class_name: str = "chessboard"
    save_visualizations: bool = True
    resume: bool = True
    seed: int = 42

    def __post_init__(self):
        """Validate configuration fields."""
        if not self.images_dir:
            raise ValueError("images_dir must be specified.")
        if not self.output_dir:
            raise ValueError("output_dir must be specified.")
        if not self.model_path:
            raise ValueError("model_path must be specified.")


def draw_annotation_debug(image: np.ndarray, corners: np.ndarray, class_name: str) -> np.ndarray:
    """Draw annotated 4-corner polygon and vertex labels on the image for visual verification."""
    annotated = image.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    labels = ["TL", "TR", "BR", "BL"]
    for idx, pt in enumerate(corners):
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(annotated, (px, py), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            annotated,
            labels[idx],
            (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"Class: {class_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def auto_annotate(cfg: AutoAnnotateConfig) -> None:
    """Auto-annotate images using BoardSegmentor RANSAC corners and save YOLO segmentation dataset."""
    images_dir = Path(cfg.images_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Input images directory not found: {images_dir}")

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
    img_paths: list[Path] = []
    for ext in extensions:
        img_paths.extend(images_dir.glob(ext))
    img_paths = sorted(list(set(img_paths)))

    if not img_paths:
        raise FileNotFoundError(f"No image files found in directory: {images_dir}")

    # Randomly sample images if image_count is specified
    if 0 < cfg.image_count < len(img_paths):
        print(f"[+] Randomly sampling {cfg.image_count}/{len(img_paths)} images (seed={cfg.seed})")
        random.seed(cfg.seed)
        img_paths = sorted(random.sample(img_paths, cfg.image_count))
    else:
        print(f"[+] Processing all {len(img_paths)} images from {images_dir}")

    # Setup output directory structure
    out_images_dir = output_dir / "images"
    out_labels_dir = output_dir / "labels"
    out_debug_dir = output_dir / "visual_debug"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_visualizations:
        out_debug_dir.mkdir(parents=True, exist_ok=True)

    # Initialize board segmentor
    print(f"[+] Initializing BoardSegmentor with model: {cfg.model_path}")
    segmentor = BoardSegmentor(
        model_path=cfg.model_path,
        device=cfg.device,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        imgsz=cfg.imgsz,
    )

    success_count = 0
    failed_count = 0

    for idx, img_path in enumerate(img_paths, 1):
        dest_img_path = out_images_dir / img_path.name
        dest_label_path = out_labels_dir / f"{img_path.stem}.txt"

        # Skip already annotated images if resume is enabled
        if cfg.resume and dest_label_path.exists() and dest_img_path.exists():
            print(f"[i] [{idx}/{len(img_paths)}] Skipped (already annotated): {img_path.name}")
            success_count += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[-] [{idx}/{len(img_paths)}] Failed to read image: {img_path.name}")
            failed_count += 1
            continue

        h, w = img.shape[:2]
        corners, _ = segmentor.segment_board(img, ordering="clockwise")

        if corners is None or len(corners) != 4:
            print(f"[-] [{idx}/{len(img_paths)}] No chessboard detected in: {img_path.name}")
            failed_count += 1
            continue

        # Format normalized polygon vertices (x1 y1 x2 y2 x3 y3 x4 y4)
        normalized_coords = []
        for pt in corners:
            nx = float(np.clip(pt[0] / w, 0.0, 1.0))
            ny = float(np.clip(pt[1] / h, 0.0, 1.0))
            normalized_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

        label_line = f"{cfg.class_id} " + " ".join(normalized_coords) + "\n"

        # Save image and label file
        shutil.copy2(img_path, dest_img_path)
        with open(dest_label_path, "w", encoding="utf-8") as f:
            f.write(label_line)

        # Optional visual debug image
        if cfg.save_visualizations:
            vis_img = draw_annotation_debug(img, corners, cfg.class_name)
            cv2.imwrite(str(out_debug_dir / f"{img_path.stem}_annotated.jpg"), vis_img)

        success_count += 1
        print(f"[✓] [{idx}/{len(img_paths)}] Annotated: {img_path.name}")

    # Generate dataset.yaml
    yaml_data = {
        "path": str(output_dir),
        "train": "images",
        "val": "images",
        "names": {cfg.class_id: cfg.class_name},
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print("\n" + "=" * 50)
    print(f"[✓] Annotation finished! Successfully annotated: {success_count}/{len(img_paths)} (Failed: {failed_count})")
    print(f"[✓] YOLO Dataset saved at: {output_dir}")
    print(f"[✓] Dataset YAML created at: {yaml_path}")
    print("=" * 50)


@draccus.wrap(config_path="/home/duykhongcay/hailo_ws/chess_pieces_detection/configs/auto_annotate_config.yaml")
def main(cfg: AutoAnnotateConfig) -> None:
    """Main CLI entrypoint."""
    auto_annotate(cfg)


if __name__ == "__main__":
    main()
