import math
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np

# FEN character class names mapping for chess piece detection (0..5: Black pieces, 6..11: White pieces)
FEN_CLASS_NAMES = {
    0: "B",
    1: "K",
    2: "N",
    3: "P",
    4: "Q",
    5: "R",
    6: "b",
    7: "k",
    8: "n",
    9: "p",
    10: "q",
    11: "r",
}


class ChessPieceDetector:
    """Unified detector supporting both PyTorch (.pt) via Ultralytics and Hailo NPU (.hef) via HailoRT."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        vdevice: Any = None,
    ):
        """Initialize object detector for PyTorch (.pt) or Hailo (.hef) model."""
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.vdevice = vdevice
        self.is_hef = self.model_path.suffix.lower() == ".hef"

        if self.is_hef:
            self._init_hailo()
        else:
            self._init_ultralytics()

    def _init_ultralytics(self):
        """Initialize PyTorch model via Ultralytics API."""
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))

    def _init_hailo(self):
        """Initialize Hailo NPU inference session using HailoRT SDK."""
        try:
            from hailo_platform import (
                HEF,
                VDevice,
                HailoStreamInterface,
                ConfigureParams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
            )
        except ImportError as e:
            raise ImportError(
                "hailo_platform is required to run .hef models on Hailo NPU hardware."
            ) from e

        self.hef = HEF(str(self.model_path))
        if self.vdevice is None:
            self.vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()

        self.input_params = InputVStreamParams.make(self.network_group, quantized=False, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(self.network_group, quantized=False)

    def predict(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run object detection inference on input image."""
        if self.is_hef:
            return self._predict_hef(image_bgr)
        return self._predict_ultralytics(image_bgr)

    def _predict_ultralytics(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run PyTorch YOLO inference."""
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        names = results.names
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = str(names.get(cls_id, FEN_CLASS_NAMES.get(cls_id, f"class_{cls_id}")))
                detections.append({
                    "box": xyxy,
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })
        return detections

    def _predict_hef(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run Hailo NPU inference and extract Hailo End-to-End NMS output layer tensors."""
        from hailo_platform import InferVStreams

        img_h, img_w = image_bgr.shape[:2]
        target_h, target_w = self.input_info.shape[0], self.input_info.shape[1]
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (target_w, target_h))
        input_data = {self.input_info.name: np.expand_dims(resized_img, axis=0)}

        with InferVStreams(self.network_group, self.input_params, self.output_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                raw_outputs = infer_pipeline.infer(input_data)

        detections = []
        for out_name, out_tensor in raw_outputs.items():
            # Hailo NMS outputs per-class list of detections (batch, num_classes)
            class_list = out_tensor[0] if isinstance(out_tensor, list) and len(out_tensor) > 0 and isinstance(out_tensor[0], list) else out_tensor

            for cls_id, cls_boxes in enumerate(class_list):
                cls_boxes_arr = np.asarray(cls_boxes)
                if cls_boxes_arr.size == 0:
                    continue
                if cls_boxes_arr.ndim == 1:
                    cls_boxes_arr = np.expand_dims(cls_boxes_arr, axis=0)

                for row in cls_boxes_arr:
                    if len(row) < 5:
                        continue
                    c_val = float(row[4])
                    if c_val < self.conf:
                        continue

                    ymin, xmin, ymax, xmax = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    if max(ymin, xmin, ymax, xmax) <= 1.01:
                        xmin_px = xmin * img_w
                        ymin_px = ymin * img_h
                        xmax_px = xmax * img_w
                        ymax_px = ymax * img_h
                    else:
                        scale_x = img_w / target_w
                        scale_y = img_h / target_h
                        xmin_px = xmin * scale_x
                        ymin_px = ymin * scale_y
                        xmax_px = xmax * scale_x
                        ymax_px = ymax * scale_y

                    cls_name = FEN_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    detections.append({
                        "box": np.array([xmin_px, ymin_px, xmax_px, ymax_px]),
                        "conf": c_val,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    })

        return detections


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

        xmin = max(0, min(img_w - 1, int(box[0])))
        ymin = max(0, min(img_h - 1, int(box[1])))
        xmax = max(0, min(img_w - 1, int(box[2])))
        ymax = max(0, min(img_h - 1, int(box[3])))

        if xmax > xmin and ymax > ymin:
            crop = image_bgr[ymin:ymax, xmin:xmax]
        else:
            crop = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

        crop_resized = cv2.resize(crop, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

        cell_h = cell_size + label_height
        cell = np.full((cell_h, cell_size, 3), 40, dtype=np.uint8)
        cell[0:cell_size, 0:cell_size] = crop_resized

        label_text = f"{class_name} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

        text_x = max(2, (cell_size - text_w) // 2)
        text_y = cell_size + ((label_height + text_h) // 2) - 2

        cv2.putText(cell, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        patches.append(cell)

    num_patches = len(patches)
    cols = min(max_cols, num_patches)
    rows = math.ceil(num_patches / cols)

    cell_h = cell_size + label_height
    grid_w = cols * cell_size + (cols + 1) * margin
    grid_h = rows * cell_h + (rows + 1) * margin

    grid = np.full((grid_h, grid_w, 3), 20, dtype=np.uint8)
    for idx, patch in enumerate(patches):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_size + margin)
        y = margin + r * (cell_h + margin)
        grid[y : y + cell_h, x : x + cell_size] = patch

    return grid

