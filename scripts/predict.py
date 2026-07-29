"""Inference script for chess piece detection, FEN generation, perspective transformation, and 4-panel 2x2 visualization."""

from dataclasses import dataclass
from pathlib import Path
import cv2
from PIL import Image
import draccus
from ultralytics import YOLO

from src.board_segmentor import BoardSegmentor, draw_extracted_squares
from src.object_detector import crop_and_stitch_detections
from src.result_visualizer import (
    map_detections_to_perspective_fen,
    draw_segmentation_and_lines,
    draw_yolo_detections,
    render_2d_board,
    create_2x2_visualizer,
    create_board_segment_visualizer,
)


@dataclass
class PredictConfig:
    # Path to input image file or directory containing test images
    source: str
    # Path to fine-tuned YOLO piece detection model weights file (.pt)
    model: str
    # Directory where prediction visualizer results will be saved
    output_dir: str
    # Confidence threshold for object detection
    conf: float = 0.25
    # NMS IoU threshold for object detection
    iou: float = 0.45
    # Image size for model prediction
    imgsz: int = 640
    # CUDA device ID (e.g. '0') or 'cpu'
    device: str = "cpu"
    # Board segmentor backend: 'local' (YOLO) or 'roboflow'
    seg_source_type: str = "local"
    # Path to local segmentation model (.pt) or Roboflow model ID
    seg_model: str = "chessboard-segmentation/1"
    # Roboflow API key
    roboflow_api_key: str = ""
    # Roboflow server API URL
    roboflow_api_url: str = "https://serverless.roboflow.com"
    # If True, run object detection only and output cropped bbox grid
    object_detect_only: bool = False
    # If True, run board segmentation only and output 1x2 visualization (Panels 1 & 2)
    board_segment_only: bool = False


@draccus.wrap()
def main(cfg: PredictConfig):
    """Main execution loop for chess piece detection, perspective transformation, and 2x2 grid visualization."""
    model_path = Path(cfg.model)
    source_path = Path(cfg.source)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Chess Piece Detection Visualizer ===")
    print(f"Piece detection model: {model_path}")
    print(f"Object detect only:    {cfg.object_detect_only}")
    print(f"Board segment only:    {cfg.board_segment_only}")
    if not cfg.object_detect_only:
        print(f"Board segmentor mode:  {cfg.seg_source_type} ({cfg.seg_model})")
    print(f"Input source:          {source_path}")
    print(f"Output directory:      {out_dir}")

    # Initialize BoardSegmentor only when needed
    board_segmentor = None
    if not cfg.object_detect_only:
        if cfg.seg_source_type.lower() == "local":
            board_segmentor = BoardSegmentor(
                source_type="local",
                model_path=Path(cfg.seg_model),
                device=cfg.device,
            )
        else:
            board_segmentor = BoardSegmentor(
                source_type="roboflow",
                model_id=cfg.seg_model if "/" in cfg.seg_model else "chessboard-segmentation/1",
                api_url=cfg.roboflow_api_url,
                api_key=cfg.roboflow_api_key,
            )

    # Load fine-tuned YOLO piece detection model only when needed
    yolo_model = None
    if not cfg.board_segment_only:
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

        # Board segment only mode: run segmentation & 64-square extraction only
        if cfg.board_segment_only:
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

            visualizer_rgb = create_board_segment_visualizer(
                seg_lines_bgr, extracted_sq_bgr, img_file.name
            )
            save_img_path = out_dir / f"board_segmentation_{img_file.stem}.png"
            Image.fromarray(visualizer_rgb).save(save_img_path)
            print(f"  Saved board segment visualizer to: {save_img_path}")
            continue

        # Step: YOLO piece detection
        results = yolo_model.predict(
            source=str(img_file),
            imgsz=cfg.imgsz,
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
                cls_name = str(names.get(cls_id, f"class_{cls_id}"))
                detections.append({
                    "box": xyxy,
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        # Object detect only mode: crop & stitch detected bboxes
        if cfg.object_detect_only:
            stitched_bgr = crop_and_stitch_detections(raw_bgr, detections)
            save_crop_path = out_dir / f"object_detection_crops_{img_file.stem}.png"
            cv2.imwrite(str(save_crop_path), stitched_bgr)
            print(f"  Saved cropped detections grid to: {save_crop_path}")
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

    print("\n=== Inference Pipeline Completed ===")


if __name__ == "__main__":
    main()
