import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import TQDM, colorstr


class ChessREDDataset(YOLODataset):
    """Custom YOLODataset class that reads ChessRED annotations.json directly in memory.
    
    This avoids generating thousands of intermediate YOLO .txt label files on disk,
    maintaining annotations.json as the single source of truth while leveraging 
    Ultralytics' dataset cache and data augmentation pipeline.
    """

    def __init__(self, *args, json_file="", split="train", **kwargs):
        """Initialize the ChessRED dataset instance.
        
        Args:
            json_file (str or Path): Path to annotations.json file.
            split (str): Dataset split ('train', 'val', or 'test').
        """
        self.json_file = str(json_file)
        self.split = split
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """Override image path discovery.
        
        Image file paths are resolved directly from annotations.json, 
        rather than scanning directory trees.
        """
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse ChessRED COCO JSON annotations, normalize bounding boxes, and save to a .cache file."""
        x = {"labels": []}

        if not Path(self.json_file).exists():
            raise FileNotFoundError(f"Annotations JSON file not found: {self.json_file}")

        with open(self.json_file, "r") as f:
            coco = json.load(f)

        # Filter out any non-piece category (such as 'empty') and sort remaining piece categories by id
        valid_cats = [c for c in coco.get("categories", []) if c.get("name", "") != "empty"]
        valid_cats = sorted(valid_cats, key=lambda c: c["id"])

        # Map original category id to 0-indexed contiguous class index
        cat_id_to_cls = {cat["id"]: i for i, cat in enumerate(valid_cats)}

        # Get list of image IDs belonging to current split
        splits = coco.get("splits", {})
        if self.split in splits:
            split_img_ids = set(splits[self.split]["image_ids"])
        else:
            # Fallback: use all image IDs if split is missing or not specified
            split_img_ids = {img["id"] for img in coco.get("images", [])}

        # Group piece annotations by image ID
        img_to_anns = defaultdict(list)
        anns_data = coco.get("annotations", {})
        pieces_list = anns_data.get("pieces", []) if isinstance(anns_data, dict) else (anns_data if isinstance(anns_data, list) else [])

        for ann in pieces_list:
            if ann.get("image_id") in split_img_ids:
                img_to_anns[ann["image_id"]].append(ann)

        # Process each image entry
        for img_info in TQDM(coco.get("images", []), desc=f"Parsing {self.split} annotations"):
            img_id = img_info["id"]
            if img_id not in split_img_ids:
                continue

            h, w = img_info["height"], img_info["width"]
            
            # Resolve image file path relative to json_file parent or img_path
            rel_path = img_info.get("path", img_info.get("file_name", ""))
            json_dir = Path(self.json_file).resolve().parent
            
            im_file = json_dir / rel_path
            if not im_file.exists():
                im_file = Path(self.img_path) / rel_path
            if not im_file.exists():
                im_file = Path(self.img_path).parent / rel_path
            if not im_file.exists() and rel_path.startswith("images/"):
                im_file = Path(self.img_path) / rel_path[len("images/"):]
            if not im_file.exists():
                continue

            # Read actual image dimensions from disk (fast header check via PIL)
            try:
                with Image.open(im_file) as img:
                    actual_w, actual_h = img.size
            except Exception:
                actual_w, actual_h = w, h

            self.im_files.append(str(im_file))
            bboxes = []

            for ann in img_to_anns.get(img_id, []):
                cat_id = ann.get("category_id")
                if cat_id not in cat_id_to_cls:
                    continue

                cls_idx = cat_id_to_cls[cat_id]

                # Bbox format in ChessRED: [x_top_left, y_top_left, width, height] (pixels)
                # Safely retrieve bbox and check if it exists and contains 4 coordinates
                bbox_raw = ann.get("bbox")
                if bbox_raw is None or len(bbox_raw) != 4:
                    continue

                box = np.array(bbox_raw, dtype=np.float32)
                
                # Convert top-left to center coordinates: [x_center, y_center, width, height]
                box[:2] += box[2:] / 2.0

                # Normalize coordinates by original image dimensions from JSON
                box[[0, 2]] /= w
                box[[1, 3]] /= h

                # Validate bounding box dimensions
                if box[2] <= 0 or box[3] <= 0 or box[0] < 0 or box[1] < 0:
                    continue

                bboxes.append([cls_idx, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (actual_h, actual_w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        x["hash"] = get_hash([self.json_file, str(self.img_path), self.split])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Retrieve labels from cache file if existing and valid, or trigger parsing."""
        cache_path = Path(self.json_file).parent / f"chessred_{self.split}.cache"
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path), self.split])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]


class ChessREDTrainer(DetectionTrainer):
    """Custom DetectionTrainer class integrating ChessREDDataset into Ultralytics YOLO training loop."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build and return a ChessREDDataset instance for the specified mode (train, val, or test)."""
        json_file = self.data.get("annotations_json", "datasets/annotations.json")
        return ChessREDDataset(
            img_path=img_path,
            json_file=json_file,
            split=mode,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
