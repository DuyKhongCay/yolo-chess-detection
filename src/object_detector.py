import math
from typing import Any, Dict, List
import cv2
import numpy as np


def crop_and_stitch_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    max_cols: int = 8,
    cell_size: int = 120,
    label_height: int = 30,
    margin: int = 10,
) -> np.ndarray:
    """Crop detected bounding boxes, draw label below each crop, and stitch into a grid image."""
    if not detections:
        # Return blank info image if no detections found
        blank = np.full((200, 600, 3), 30, dtype=np.uint8)
        cv2.putText(
            blank,
            "No object detections found",
            (50, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return blank

    img_h, img_w = image_bgr.shape[:2]
    patches = []

    for det in detections:
        box = det["box"]
        class_name = det.get("class_name", "object")
        conf = det.get("conf", 0.0)

        # Clip box coordinates safely
        xmin = max(0, min(img_w - 1, int(box[0])))
        ymin = max(0, min(img_h - 1, int(box[1])))
        xmax = max(0, min(img_w - 1, int(box[2])))
        ymax = max(0, min(img_h - 1, int(box[3])))

        # Crop patch or handle degenerate box
        if xmax > xmin and ymax > ymin:
            crop = image_bgr[ymin:ymax, xmin:xmax]
        else:
            crop = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

        # Resize crop image patch
        crop_resized = cv2.resize(crop, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

        # Create individual cell canvas with label bar underneath
        cell_h = cell_size + label_height
        cell = np.full((cell_h, cell_size, 3), 40, dtype=np.uint8)
        cell[0:cell_size, 0:cell_size] = crop_resized

        # Draw label text below bbox crop
        label_text = f"{class_name} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Center text in label bar area
        text_x = max(2, (cell_size - text_w) // 2)
        text_y = cell_size + ((label_height + text_h) // 2) - 2

        cv2.putText(cell, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        patches.append(cell)

    # Calculate grid dimensions
    num_patches = len(patches)
    cols = min(max_cols, num_patches)
    rows = math.ceil(num_patches / cols)

    cell_h = cell_size + label_height
    grid_w = cols * cell_size + (cols + 1) * margin
    grid_h = rows * cell_h + (rows + 1) * margin

    # Stitch patches into grid composite image
    grid = np.full((grid_h, grid_w, 3), 20, dtype=np.uint8)
    for idx, patch in enumerate(patches):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_size + margin)
        y = margin + r * (cell_h + margin)
        grid[y : y + cell_h, x : x + cell_size] = patch

    return grid
