"""Result visualization and FEN mapping utilities for chess piece detection."""

import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Unicode symbols for rendering 2D board representation
UNICODE_PIECES = {
    "P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔",
    "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚",
}

# Color palette for drawing YOLO bounding boxes (BGR format)
CLASS_COLORS = {
    "P": (255, 200, 100), 
    "R": (255, 170, 0),
    "N": (255, 140, 0), 
    "B": (255, 100, 0),
    "Q": (255, 50, 0), 
    "K": (255, 0, 100),
    "p": (100, 200, 255),
    "r": (0, 170, 255), 
    "n": (0, 140, 255), 
    "b": (0, 100, 255), 
    "q": (50, 0, 255), 
    "k": (100, 0, 255),
}


def map_detections_to_perspective_fen(detections, cell_dict):
    """Map YOLO piece detections to the 64 perspective-warped chessboard squares and build a FEN string.
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
        # Apply upward offset
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
            # Calculate maximum distance from center to corners of the cell
            max_r = max(math.hypot(c_pt[0] - cx, c_pt[1] - cy) for c_pt in closest_info["points"])
            # Allow fallback mapping only if the distance is within 1.3 times the cell radius
            if best_dist > max_r * 1.3:
                best_cell = None

        # Get FEN character directly from class_name
        cls_name = det.get("class_name")
        cls_id = det.get("class_id")
        fen_char = cls_name

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
                    f"-> WARNING: FEN character lookup failed!"
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
    """
    annotated = image.copy()
    if debug_info is None:
        return annotated

    h, w = annotated.shape[:2]

    # Draw segmentation polygon mask
    poly = debug_info.get("polygon")
    if poly is not None and len(poly) > 0:
        mask = np.zeros_like(annotated)
        cv2.fillPoly(mask, [poly.astype(np.int32)], (0, 255, 0))
        cv2.addWeighted(mask, 0.25, annotated, 0.75, 0, annotated)
        cv2.polylines(annotated, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw 4 fitted RANSAC edge lines
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
        length = max(h, w) * 2
        p1 = (int(x0 - vx * length), int(y0 - vy * length))
        p2 = (int(x0 + vx * length), int(y0 + vy * length))
        color = line_colors[idx % len(line_colors)]
        cv2.line(annotated, p1, p2, color, 2)

    # Draw 4 corner points
    if corners is not None:
        for pt in corners:
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 8, (0, 165, 255), -1)
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 8, (0, 255, 255), 2)

    return annotated


def create_board_segment_visualizer(seg_img_bgr, extracted_sq_bgr, img_name):
    """Create a 1x2 composite visualizer image showing board segmentation and extracted squares."""
    seg_img_rgb = cv2.cvtColor(seg_img_bgr, cv2.COLOR_BGR2RGB)
    extracted_sq_rgb = cv2.cvtColor(extracted_sq_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    fig.suptitle(
        f"Chessboard Segmentation Pipeline: {img_name}",
        fontsize=11, fontweight="bold", y=0.98
    )

    # Panel 1: Segmentation & RANSAC Lines
    axes[0].set_title("1. Segmentation & RANSAC Lines", fontsize=11, fontweight="bold", pad=8)
    axes[0].imshow(seg_img_rgb)
    axes[0].axis("off")

    # Panel 2: Extracted Squares
    axes[1].set_title("2. Extracted Squares", fontsize=11, fontweight="bold", pad=8)
    axes[1].imshow(extracted_sq_rgb)
    axes[1].axis("off")

    plt.tight_layout()
    composite_rgb = canvas_to_numpy_rgb(fig)
    plt.close(fig)

    return composite_rgb


def create_2x2_visualizer(seg_img_bgr, extracted_sq_bgr, bbox_bgr, board_2d_rgb, img_name, fen_str):
    """Create a 4-panel 2x2 grid composite visualizer image.

    Args:
        seg_img_bgr (np.ndarray): Segmentation & RANSAC lines image (BGR).
        extracted_sq_bgr (np.ndarray): Image with extracted 64-square grid polygons (BGR).
        bbox_bgr (np.ndarray): Image with YOLO bounding boxes (BGR).
        board_2d_rgb (np.ndarray): Rendered 2D board image (RGB).
        img_name (str): Input filename.
        fen_str (str): Generated FEN string.

    Returns:
        np.ndarray: Composite RGB visualizer image.
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


def create_object_detection_visualizer(bbox_bgr: np.ndarray, stitched_bgr: np.ndarray, img_name: str) -> np.ndarray:
    """Create a 1x2 composite visualizer image showing bounding box detections and cropped objects grid."""
    bbox_rgb = cv2.cvtColor(bbox_bgr, cv2.COLOR_BGR2RGB)
    stitched_rgb = cv2.cvtColor(stitched_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    fig.suptitle(
        f"Object Detection Pipeline: {img_name}",
        fontsize=11, fontweight="bold", y=0.98
    )

    # Panel 1: Original Image with Bounding Boxes
    axes[0].set_title("1. Piece Bounding Box Detections", fontsize=11, fontweight="bold", pad=8)
    axes[0].imshow(bbox_rgb)
    axes[0].axis("off")

    # Panel 2: Cropped Detections Grid
    axes[1].set_title("2. Cropped Objects Grid", fontsize=11, fontweight="bold", pad=8)
    axes[1].imshow(stitched_rgb)
    axes[1].axis("off")

    plt.tight_layout()
    composite_rgb = canvas_to_numpy_rgb(fig)
    plt.close(fig)

    return composite_rgb


def canvas_to_numpy_rgb(fig):
    """Convert matplotlib figure canvas to a numpy RGB array."""
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
