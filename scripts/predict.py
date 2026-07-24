"""Inference script for chess piece detection, FEN generation, perspective transformation, and 4-panel 2x2 visualization."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root directory to path to allow importing modules inside scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import draccus
from ultralytics import YOLO

from src.board_segmentor import (
    BoardSegmentor,
    draw_extracted_squares,
    extract_chessboard_perspective,
)

# Mapping of YOLO class names to FEN characters
CLASS_NAME_TO_FEN = {
    # "white-pawn": "P",
    # "white-knight": "N",
    # "white-bishop": "B",
    # "white-rook": "R",
    # "white-queen": "Q",
    # "white-king": "K",
    # "black-pawn": "p",
    # "black-knight": "n",
    # "black-bishop": "b",
    # "black-rook": "r",
    # "black-queen": "q",
    # "black-king": "k", 
    # "white-pawn": "P",
    0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",
    6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",
}
# Unicode symbols for rendering 2D board representation
UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

# Color palette for drawing YOLO bounding boxes (BGR format)
CLASS_COLORS = {
    "white-pawn": (255, 200, 100),
    "white-knight": (255, 170, 0),
    "white-bishop": (255, 140, 0),
    "white-rook": (255, 100, 0),
    "white-queen": (255, 50, 0),
    "white-king": (255, 0, 100),
    "black-pawn": (100, 200, 255),
    "black-knight": (0, 170, 255),
    "black-bishop": (0, 140, 255),
    "black-rook": (0, 100, 255),
    "black-queen": (50, 0, 255),
    "black-king": (100, 0, 255),
}


@dataclass
class PredictConfig:
    """Dataclass configuration for chess piece inference and visualization."""

    source: str = field(
        default=str(PROJECT_ROOT / "datasets" / "images"),
        metadata={"help": "Path to input image file or directory containing test images."},
    )
    model: str = field(
        default=str(
            PROJECT_ROOT / "runs" / "detect" / "models" / "yolo11m_chessred" / "weights" / "best.pt"
        ),
        metadata={"help": "Path to fine-tuned YOLO piece detection model weights file (.pt)."},
    )
    output_dir: str = field(
        default=str(PROJECT_ROOT / "runs" / "predict"),
        metadata={"help": "Directory where prediction visualizer results will be saved."},
    )
    conf: float = field(
        default=0.25,
        metadata={"help": "Confidence threshold for object detection."},
    )
    iou: float = field(
        default=0.45,
        metadata={"help": "NMS IoU threshold for object detection."},
    )
    device: str = field(
        default="cpu",
        metadata={"help": "CUDA device ID (e.g. '0') or 'cpu'."},
    )

    # Board Segmentor Configuration
    seg_source_type: str = field(
        default="local",
        metadata={"help": "Board segmentor backend: 'local' (YOLO) or 'roboflow'."},
    )
    seg_model: str = field(
        default="chessboard-segmentation/1",
        metadata={
            "help": "Path to local segmentation model (.pt) or Roboflow model ID (e.g. 'chessboard-segmentation/1')."
        },
    )
    roboflow_api_key: str = field(
        default="",
        metadata={"help": "Roboflow API key."},
    )
    roboflow_api_url: str = field(
        default="https://serverless.roboflow.com",
        metadata={"help": "Roboflow server API URL."},
    )


def map_detections_to_perspective_fen(detections, cell_dict):
    """Map YOLO piece detections to the 64 perspective-warped chessboard squares and build a FEN string.

    Args:
        detections (list): List of detection dicts with 'box', 'class_name', 'conf'.
        cell_dict (dict): Dictionary of 64 cell polygon definitions.

    Returns:
        str: FEN position string.
    """
    cell_pieces = {cell_id: None for cell_id in range(1, 65)}
    cell_confs = {cell_id: -1.0 for cell_id in range(1, 65)}

    # Base offset ratio to shift y coordinate slightly upward from bbox bottom (y2)
    # to prevent the piece base from spilling into the square closer to the camera.
    base_offset_ratio = 0.3

    print(f"\n--- [DEBUG FEN MAPPER] Processing {len(detections)} detected piece(s) (offset_ratio={base_offset_ratio}) ---")

    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["box"]
        box_h = float(y2 - y1)
        x_mid = float((x1 + x2) / 2.0)
        # Apply upward offset of 15% of bbox height
        y_base = float(y2 - (box_h * base_offset_ratio))
        pt = (x_mid, y_base)

        best_cell = None
        best_dist = float("inf")
        is_inside_any = False

        for cell_id, info in cell_dict.items():
            poly = info["polygon"]
            inside = cv2.pointPolygonTest(poly, pt, False)
            if inside >= 0:
                best_cell = cell_id
                is_inside_any = True
                break

            cx, cy = info["center"]
            dist = math.hypot(x_mid - cx, y_base - cy)
            if dist < best_dist:
                best_dist = dist
                best_cell = cell_id

        # If not inside any cell, perform threshold check on the closest cell distance
        if not is_inside_any and best_cell is not None:
            closest_info = cell_dict[best_cell]
            cx, cy = closest_info["center"]
            # Calculate maximum distance from center to corners of the cell (local cell radius)
            max_r = max(math.hypot(c_pt[0] - cx, c_pt[1] - cy) for c_pt in closest_info["points"])
            # Allow fallback mapping only if the distance is within 1.3 times the cell radius
            if best_dist > max_r * 1.3:
                best_cell = None

        # Lookup FEN character trying class_name, class_id, and str(class_id)
        cls_name = det.get("class_name")
        cls_id = det.get("class_id")
        fen_char = (
            CLASS_NAME_TO_FEN.get(cls_name)
            or CLASS_NAME_TO_FEN.get(cls_id)
            or CLASS_NAME_TO_FEN.get(str(cls_id) if cls_id is not None else "")
        )

        if best_cell is not None:
            if fen_char:
                if det["conf"] > cell_confs[best_cell]:
                    old_piece = cell_pieces[best_cell]
                    cell_pieces[best_cell] = fen_char
                    cell_confs[best_cell] = det["conf"]
                    replace_info = f" (replaced '{old_piece}')" if old_piece else ""
                    print(
                        f"  Det #{idx:02d}: cls_id={cls_id}, cls_name='{cls_name}', conf={det['conf']:.2f} "
                        f"-> mapped to Cell #{best_cell:02d} as '{fen_char}'{replace_info}"
                    )
                else:
                    print(
                        f"  Det #{idx:02d}: cls_id={cls_id}, cls_name='{cls_name}', conf={det['conf']:.2f} "
                        f"-> Cell #{best_cell:02d} already has higher conf piece '{cell_pieces[best_cell]}'"
                    )
            else:
                print(
                    f"  Det #{idx:02d}: cls_id={cls_id}, cls_name='{cls_name}', conf={det['conf']:.2f} "
                    f"-> WARNING: FEN character lookup failed! Not found in CLASS_NAME_TO_FEN."
                )
        else:
            print(
                f"  Det #{idx:02d}: cls_id={cls_id}, cls_name='{cls_name}', conf={det['conf']:.2f} "
                f"-> IGNORED (outside chessboard boundary)"
            )

    # Print summary of mapped pieces per cell
    placed_pieces = {cid: p for cid, p in cell_pieces.items() if p is not None}
    print(f"--- [DEBUG FEN MAPPER SUMMARY] Total pieces mapped: {len(placed_pieces)}/64 ---")
    if placed_pieces:
        pieces_str = ", ".join([f"Cell {cid}: '{p}'" for cid, p in sorted(placed_pieces.items())])
        print(f"  Placed pieces: {pieces_str}")

    fen_ranks = []
    for rank_idx in range(7, -1, -1):  # Rank 8 down to Rank 1
        rank_str = ""
        empty_count = 0
        start_cell = rank_idx * 8 + 1
        for file_idx in range(8):
            cell_id = start_cell + file_idx
            piece = cell_pieces[cell_id]
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece
        if empty_count > 0:
            rank_str += str(empty_count)
        fen_ranks.append(rank_str)

    fen_body = "/".join(fen_ranks)
    full_fen = f"{fen_body} w - - 0 1"
    print(f"--- [DEBUG FEN RESULT] {full_fen} ---\n")
    return full_fen


def render_2d_board(fen_str):
    """Render a 2D chessboard visual representation from a FEN string using Matplotlib.

    Args:
        fen_str (str): FEN position string.

    Returns:
        np.ndarray: RGB image array of rendered 2D chessboard.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    light_color = "#F0D9B5"
    dark_color = "#B58863"

    for r in range(8):
        for c in range(8):
            color = light_color if (r + c) % 2 == 0 else dark_color
            rect = plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="none")
            ax.add_patch(rect)

    fen_position = fen_str.split()[0]
    ranks = fen_position.split("/")

    for rank_idx, rank_str in enumerate(ranks):
        r = 7 - rank_idx
        c = 0
        for char in rank_str:
            if char.isdigit():
                c += int(char)
            else:
                piece_symbol = UNICODE_PIECES.get(char, char)
                text_color = "white" if char.isupper() else "black"
                if char.isupper():
                    ax.text(
                        c + 0.5, r + 0.48, piece_symbol,
                        fontsize=28, ha="center", va="center",
                        color="black", fontweight="bold", alpha=0.5
                    )
                ax.text(
                    c + 0.5, r + 0.5, piece_symbol,
                    fontsize=28, ha="center", va="center",
                    color=text_color, fontweight="bold"
                )
                c += 1

    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for c in range(8):
        ax.text(c + 0.5, 0.15, files[c], fontsize=10, ha="center", va="center", color="#404040", fontweight="bold")
    for r in range(8):
        ax.text(0.15, r + 0.85, str(r + 1), fontsize=10, ha="center", va="center", color="#404040", fontweight="bold")

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    image_grid = canvas_to_numpy_rgb(fig)
    plt.close(fig)

    return image_grid


def draw_yolo_detections(image, detections):
    """Draw YOLO bounding boxes, class labels, confidence scores, and base reference points on image.

    Args:
        image (np.ndarray): Original BGR image.
        detections (list): List of detection dictionaries.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = image.copy()
    base_offset_ratio = 0.15

    for det in detections:
        x1, y1, x2, y2 = map(float, det["box"])
        cls_name = det["class_name"]
        conf = det["conf"]

        box_h = y2 - y1
        x_mid = (x1 + x2) / 2.0
        y_base = y2 - (box_h * base_offset_ratio)

        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw red dot at calculated piece base reference point
        cv2.circle(annotated, (int(x_mid), int(y_base)), 4, (0, 0, 255), -1)

        label = f"{cls_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        label_y1 = max(int(y1) - label_h - 4, 0)
        cv2.rectangle(annotated, (int(x1), label_y1), (int(x1) + label_w + 4, label_y1 + label_h + 4), color, -1)
        cv2.putText(
            annotated, label, (int(x1) + 2, label_y1 + label_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    return annotated


def draw_segmentation_and_lines(image, corners, debug_info):
    """Draw YOLO segmentation contour, fitted RANSAC lines, and corner points on image.

    Args:
        image (np.ndarray): Original BGR image.
        corners (np.ndarray): 4 corner points of the chessboard.
        debug_info (dict): Dictionary with 'polygon' and 'lines'.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = image.copy()
    if debug_info is None:
        return annotated

    h, w = annotated.shape[:2]

    # 1. Draw segmentation polygon mask (semi-transparent overlay)
    poly = debug_info.get("polygon")
    if poly is not None and len(poly) > 0:
        mask = np.zeros_like(annotated)
        cv2.fillPoly(mask, [poly.astype(np.int32)], (0, 255, 0))
        cv2.addWeighted(mask, 0.25, annotated, 0.75, 0, annotated)
        cv2.polylines(annotated, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # 2. Draw 4 fitted RANSAC edge lines
    lines = debug_info.get("lines", [])
    line_colors = [
        (255, 0, 0),    # Top - Blue
        (0, 255, 255),  # Right - Yellow
        (0, 0, 255),    # Bottom - Red
        (255, 0, 255)   # Left - Magenta
    ]
    for idx, line in enumerate(lines):
        if line is None:
            continue
        vx, vy, x0, y0 = line
        # Project line segment to cross the entire screen boundaries
        length = max(h, w) * 2
        p1 = (int(x0 - vx * length), int(y0 - vy * length))
        p2 = (int(x0 + vx * length), int(y0 + vy * length))
        color = line_colors[idx % len(line_colors)]
        cv2.line(annotated, p1, p2, color, 2)

    # 3. Draw 4 corner points
    if corners is not None:
        for pt in corners:
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 8, (0, 165, 255), -1)  # Orange dot
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), 2)   # Yellow boundary

    return annotated


def create_2x2_visualizer(seg_img_bgr, extracted_sq_bgr, bbox_bgr, board_2d_rgb, img_name, fen_str):
    """Create a 4-panel 2x2 grid composite visualizer image ordered by processing pipeline sequence.

    Row 1, Col 1 (Top-Left):     1. Segmentation & RANSAC Lines
    Row 1, Col 2 (Top-Right):    2. Extracted Squares
    Row 2, Col 1 (Bottom-Left):  3. YOLO Bounding Box
    Row 2, Col 2 (Bottom-Right): 4. Converted Image

    Args:
        seg_img_bgr (np.ndarray): Segmentation & RANSAC lines image (BGR).
        extracted_sq_bgr (np.ndarray): Image with extracted 64-square grid polygons (BGR).
        bbox_bgr (np.ndarray): Image with YOLO bounding boxes (BGR).
        board_2d_rgb (np.ndarray): Rendered 2D board image (RGB).
        img_name (str): Input filename.
        fen_str (str): Generated FEN string.

    Returns:
        np.ndarray: Composite RGB visualizer image (2x2 grid).
    """
    seg_img_rgb = cv2.cvtColor(seg_img_bgr, cv2.COLOR_BGR2RGB)
    extracted_sq_rgb = cv2.cvtColor(extracted_sq_bgr, cv2.COLOR_BGR2RGB)
    bbox_rgb = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=120)
    fig.suptitle(
        f"Chessboard Processing & Piece Detection Pipeline: {img_name}\nFEN: {fen_str}",
        fontsize=11, fontweight="bold", y=0.98
    )

    # Panel 1 (Top-Left): 1. Segmentation & RANSAC Lines
    axes[0, 0].set_title("1. Segmentation & RANSAC Lines", fontsize=11, fontweight="bold", pad=8)
    axes[0, 0].imshow(seg_img_rgb)
    axes[0, 0].axis("off")

    # Panel 2 (Top-Right): 2. Extracted Squares
    axes[0, 1].set_title("2. Extracted Squares", fontsize=11, fontweight="bold", pad=8)
    axes[0, 1].imshow(extracted_sq_rgb)
    axes[0, 1].axis("off")

    # Panel 3 (Bottom-Left): 3. YOLO Bounding Box
    axes[1, 0].set_title("3. YOLO Bounding Box", fontsize=11, fontweight="bold", pad=8)
    axes[1, 0].imshow(bbox_rgb)
    axes[1, 0].axis("off")

    # Panel 4 (Bottom-Right): 4. Converted Image
    axes[1, 1].set_title("4. Converted Image", fontsize=11, fontweight="bold", pad=8)
    axes[1, 1].imshow(board_2d_rgb)
    axes[1, 1].axis("off")

    plt.tight_layout()
    composite_rgb = canvas_to_numpy_rgb(fig)
    plt.close(fig)

    return composite_rgb


def canvas_to_numpy_rgb(fig):
    """Convert matplotlib figure canvas to a numpy RGB array compatible across Matplotlib versions."""
    fig.canvas.draw()
    canvas = fig.canvas
    if hasattr(canvas, "buffer_rgba"):
        return np.asarray(canvas.buffer_rgba())[:, :, :3]
    elif hasattr(canvas, "tostring_rgb"):
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        return image_flat.reshape(canvas.get_width_height()[::-1] + (3,))
    elif hasattr(canvas, "tostring_argb"):
        image_flat = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        image_argb = image_flat.reshape(canvas.get_width_height()[::-1] + (4,))
        return image_argb[:, :, 1:]
    else:
        return np.asarray(canvas.buffer_argb())[:, :, 1:]


@draccus.wrap()
def main(cfg: PredictConfig):
    """Main execution loop for chess piece detection, perspective transformation, and 2x2 grid visualization."""
    # Resolve relative paths against PROJECT_ROOT
    model_path = Path(cfg.model)
    if not model_path.is_absolute():
        if (PROJECT_ROOT / model_path).exists():
            model_path = (PROJECT_ROOT / model_path).resolve()
        else:
            model_path = model_path.resolve()

    if not model_path.exists():
        for candidate in [
            PROJECT_ROOT / "runs" / "detect" / "models" / "yolo11m_chessred-5" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "detect" / "models" / "yolo11m_chessred" / "weights" / "best.pt",
            PROJECT_ROOT / "yolo11m.pt",
        ]:
            if candidate.exists():
                model_path = candidate
                break

    source_path = Path(cfg.source)
    if not source_path.is_absolute():
        if (PROJECT_ROOT / source_path).exists():
            source_path = (PROJECT_ROOT / source_path).resolve()
        else:
            source_path = source_path.resolve()

    out_dir = Path(cfg.output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Chess Piece Detection & 2x2 Pipeline Visualizer ===")
    print(f"Piece detection model: {model_path}")
    print(f"Board segmentor mode:  {cfg.seg_source_type} ({cfg.seg_model})")
    print(f"Input source:          {source_path}")
    print(f"Output directory:      {out_dir}")

    # Initialize BoardSegmentor (YOLO Local or Roboflow)
    if cfg.seg_source_type.lower() == "local":
        seg_model_path = Path(cfg.seg_model)
        if not seg_model_path.is_absolute() and (PROJECT_ROOT / seg_model_path).exists():
            seg_model_path = (PROJECT_ROOT / seg_model_path).resolve()
        board_segmentor = BoardSegmentor(
            source_type="local",
            model_path=seg_model_path,
            device=cfg.device,
        )
    else:
        board_segmentor = BoardSegmentor(
            source_type="roboflow",
            model_id=cfg.seg_model if "/" in cfg.seg_model else "chessboard-segmentation/1",
            api_url=cfg.roboflow_api_url,
            api_key=cfg.roboflow_api_key,
        )

    # Load fine-tuned YOLO piece detection model
    yolo_model = YOLO(str(model_path))

    # Resolve list of input image paths
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = sorted(
            [p for p in source_path.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        )
    else:
        print(f"Error: Source path {source_path} does not exist.")
        return

    if not image_files:
        print(f"No valid image files found in {source_path}.")
        return

    print(f"Found {len(image_files)} image(s) for processing.")

    for idx, img_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_file.name}")
        raw_bgr = cv2.imread(str(img_file))
        if raw_bgr is None:
            print(f"Warning: Failed to load image {img_file}")
            continue

        # Step 1: Segmentation & Perspective transformation & 64 chessboard squares extraction
        corners, debug_info = board_segmentor.segment_board(raw_bgr)
        if corners is None:
            print(f"Warning: Board Segmentor failed to detect chessboard corners in {img_file.name}")
            continue

        warped_board, M, M_inv, sq_orig, cell_dict = board_segmentor.extract_chessboard_perspective(
            raw_bgr, corners=corners
        )
        if cell_dict is None:
            print(f"Warning: Board Segmentor failed to extract chessboard perspective in {img_file.name}")
            continue

        seg_lines_bgr = draw_segmentation_and_lines(raw_bgr, corners, debug_info)
        extracted_sq_bgr = draw_extracted_squares(raw_bgr, cell_dict)

        # Step 2: YOLO piece detection
        results = yolo_model.predict(
            source=str(img_file),
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            verbose=False,
        )[0]

        detections = []
        names = results.names
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "box": xyxy,
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        # Step 3: Draw YOLO bounding boxes and base reference points
        bbox_bgr = draw_yolo_detections(raw_bgr, detections)

        # Step 4: Map piece detections into perspective grid & build FEN
        fen_str = map_detections_to_perspective_fen(detections, cell_dict)
        print(f"  Generated FEN: {fen_str}")

        # Step 5: Render 2D Chessboard image from FEN
        board_2d_rgb = render_2d_board(fen_str)

        # Step 6: Create 2x2 Grid 4-Panel Composite Visualizer Image
        visualizer_rgb = create_2x2_visualizer(
            seg_lines_bgr, extracted_sq_bgr, bbox_bgr, board_2d_rgb, img_file.name, fen_str
        )

        # Save composite visualizer image and FEN text file
        save_img_path = out_dir / f"perspective_transformation_result_{img_file.stem}.png"
        save_fen_path = out_dir / f"{img_file.stem}_fen.txt"

        Image.fromarray(visualizer_rgb).save(save_img_path)
        with open(save_fen_path, "w", encoding="utf-8") as f:
            f.write(fen_str + "\n")

        print(f"  Saved visualizer to: {save_img_path}")
        print(f"  Saved FEN string to: {save_fen_path}")

    print("\n=== Inference & 2x2 Grid Visualization Pipeline Completed ===")


if __name__ == "__main__":
    main()
