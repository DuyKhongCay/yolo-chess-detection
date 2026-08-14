"""Dataset conversion utility script supporting COCO-to-YOLO format and Image-to-NPY array conversion."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import shutil

import draccus
import numpy as np
from PIL import Image
import yaml


@dataclass
class CocoToYoloConfig:
    """Configuration for COCO JSON to YOLO format conversion."""

    json_path: str = ""
    images_dir: str = ""
    output_dir: str = ""
    use_split_key: bool = False
    max_images: int | None = None
    # seed: int = 42


@dataclass
class ImageToNpyConfig:
    """Configuration for converting an image directory to a NumPy NPY array."""

    images_dir: str | None = None
    output_npy: str | None = None
    imgsz: int = 640
    dtype: str = "float32"
    shape: list[int] | None = None
    num_samples: int | None = None
    seed: int = 42

    def __post_init__(self):
        """Validate path configurations immediately upon initialization."""
        if not self.images_dir:
            raise ValueError("images_dir must be specified for image_to_npy mode.")
        if not self.output_npy:
            raise ValueError("output_npy must be specified for image_to_npy mode.")
        if not Path(self.images_dir).resolve().exists():
            raise FileNotFoundError(f"Input images directory not found: {self.images_dir}")


@dataclass
class ConvertDatasetConfig:
    """Main dataclass configuration for dataset format conversion."""

    mode: str | None = None  # Options: 'coco_to_yolo', 'image_to_npy'
    coco_config: CocoToYoloConfig | None = None
    image_config: ImageToNpyConfig | None = None


def convert_image_to_npy(cfg: ImageToNpyConfig) -> None:
    """Convert directory of images to NumPy NPY array format."""
    images_dir = Path(cfg.images_dir).resolve()
    output_path = Path(cfg.output_npy).resolve()

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.JPEG", "*.PNG")
    img_paths = []
    for ext in extensions:
        img_paths.extend(images_dir.glob(ext))
    img_paths = sorted(list(set(img_paths)))

    if not img_paths:
        raise FileNotFoundError(f"No image files found in directory: {images_dir}")

    if cfg.num_samples is not None and cfg.num_samples < len(img_paths):
        print(f"[+] Randomly sampling {cfg.num_samples} images from total {len(img_paths)} (seed={cfg.seed})")
        random.seed(cfg.seed)
        img_paths = random.sample(img_paths, cfg.num_samples)
        img_paths.sort()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype(cfg.dtype)

    if cfg.shape and len(cfg.shape) >= 2:
        h, w = cfg.shape[0], cfg.shape[1]
    else:
        h, w = cfg.imgsz, cfg.imgsz

    total_imgs = len(img_paths)
    print(f"[+] Converting {total_imgs} images from {images_dir} to NPY array (shape: ({total_imgs}, {h}, {w}, 3), dtype: {dt})")

    data = np.zeros((total_imgs, h, w, 3), dtype=dt)
    for idx, img_p in enumerate(img_paths):
        with Image.open(img_p) as img:
            img_rgb = img.convert("RGB").resize((w, h))
            data[idx] = np.array(img_rgb, dtype=dt)

    np.save(output_path, data)
    print(f"[✓] Saved NPY file successfully: {output_path} (shape: {data.shape}, dtype: {data.dtype})")


def _split_coco_json_by_splits(coco_data: dict, temp_dir: Path) -> dict[str, Path]:
    """Split COCO json into train/val/test json files using the splits key."""
    categories = coco_data.get("categories", [])
    anns_raw = coco_data.get("annotations", {})
    anns_list = anns_raw.get("pieces", []) if isinstance(anns_raw, dict) else (anns_raw if isinstance(anns_raw, list) else [])
    
    # Group annotations by image_id
    img_to_anns = {}
    for ann in anns_list:
        if ann.get("bbox") and len(ann.get("bbox")) == 4:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Filter out images without annotations
    valid_images = {img["id"]: img for img in coco_data.get("images", []) if img["id"] in img_to_anns}
    splits_info = coco_data.get("splits", {})
    split_map = {"train": "train", "val": "val", "test": "test"}
    split_json_paths = {}

    for s_key, s_val in splits_info.items():
        split_name = split_map.get(s_key, s_key)
        target_img_ids = set(s_val.get("image_ids", [])) & set(valid_images.keys())
        
        split_imgs = [valid_images[i] for i in target_img_ids]
        split_anns = [ann for img_id in target_img_ids for ann in img_to_anns.get(img_id, [])]
        
        split_coco = {
            "images": split_imgs,
            "annotations": split_anns,
            "categories": categories,
        }
        
        split_json_path = temp_dir / f"annotations_{split_name}.json"
        with open(split_json_path, "w") as f:
            json.dump(split_coco, f)
        split_json_paths[split_name] = split_json_path

    return split_json_paths


def convert_coco_to_yolo(cfg: CocoToYoloConfig) -> None:
    """Convert COCO annotations to YOLO format using Ultralytics converter API."""
    from ultralytics.data.converter import convert_coco

    json_path = Path(cfg.json_path).resolve()
    output_dir = Path(cfg.output_dir).resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {json_path}")

    print(f"[+] Converting COCO to YOLO via Ultralytics API from: {json_path}")
    
    if cfg.use_split_key:
        with open(json_path, "r") as f:
            coco_data = json.load(f)
            
        temp_split_dir = output_dir / "_temp_splits"
        temp_split_dir.mkdir(parents=True, exist_ok=True)
        split_paths = _split_coco_json_by_splits(coco_data, temp_split_dir)
        
        for split_name, s_json_path in split_paths.items():
            convert_coco(labels_dir=str(s_json_path.parent), save_dir=str(output_dir / split_name), use_segments=False)
            
        shutil.rmtree(temp_split_dir, ignore_errors=True)
    else:
        convert_coco(labels_dir=str(json_path.parent), save_dir=str(output_dir), use_segments=False)

    # Clean up empty label files (unannotated images)
    deleted_count = 0
    for txt_file in output_dir.glob("**/*.txt"):
        if txt_file.stat().st_size == 0:
            txt_file.unlink()
            deleted_count += 1

    print(f"[✓] Conversion completed. Cleaned up {deleted_count} empty/unannotated label files.")


@draccus.wrap()
def main(cfg: ConvertDatasetConfig):
    """Main entrypoint for dataset format conversion."""
    if cfg.mode == "image_to_npy":
        convert_image_to_npy(cfg.image_config)
    elif cfg.mode == "coco_to_yolo":
        convert_coco_to_yolo(cfg.coco_config)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}. Supported modes: 'coco_to_yolo', 'image_to_npy'")


if __name__ == "__main__":
    main()

