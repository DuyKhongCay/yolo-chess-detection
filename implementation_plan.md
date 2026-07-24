# Kế hoạch Phát triển Dự án Chess Pieces Detection (Fine-tune YOLOv11m trực tiếp từ COCO JSON & Perspective Transformation)

Dự án này hướng tới việc xây dựng pipeline phát hiện quân cờ và căn chỉnh bàn cờ end-to-end. Dựa trên nghiên cứu tài liệu Ultralytics mới nhất trong `ok.md`, chúng ta sẽ sử dụng phương pháp **Subclassing `YOLODataset` và `DetectionTrainer`** để **huấn luyện trực tiếp mô hình YOLOv11m từ file `annotations.json` (chuẩn COCO JSON)** của ChessRED mà **không cần chuyển đổi (convert) thành hàng nghìn file nhãn `.txt` trung gian**.

---

## 1. Ưu điểm của Phương pháp Mới (Direct COCO JSON Training)

1. **Single Source of Truth**: Giữ nguyên file `annotations.json` duy nhất của ChessRED làm nguồn dữ liệu chính.
2. **Không tạo file nhãn rác**: Loại bỏ bước sinh hàng ngàn thư mục và file `.txt` trên ổ đĩa.
3. **Tự động Caching**: Ultralytics tự động parse JSON lần đầu và lưu file `.cache`. Các lần train sau sẽ load tức thì từ cache.
4. **Hỗ trợ Augmentations toàn diện**: Tận dụng đầy đủ các kỹ thuật tăng cường dữ liệu sẵn có của Ultralytics (Mosaic, Mixup, Random Scale, HSV, v.v.).

---

## 2. Cấu trúc Thư mục Dự án

```text
chess_pieces_detection/
├── datasets/
│   ├── annotations.json          # File nhãn gốc duy nhất của ChessRED
│   ├── images/                   # Thư mục ảnh gốc ChessRED
│   └── dataset.yaml              # File cấu hình dữ liệu YOLO (chỉ định đường dẫn JSON & classes)
├── scripts/
│   ├── chessred_download.py      # Tải & giải nén tập dữ liệu (đã có)
│   ├── dataset.py                # Định nghĩa ChessREDDataset (subclass YOLODataset) & ChessREDTrainer (subclass DetectionTrainer)
│   ├── preprocess_perspective.py  # Xử lý biến đổi góc nhìn bàn cờ & trích xuất 64 ô cờ (Perspective Transformation)
│   ├── train.py                   # Script fine-tune YOLOv11m với custom trainer
│   ├── evaluate.py                # Script đánh giá mô hình trên tập test (mAP50, mAP50-95)
│   └── predict.py                 # Pipeline suy luận end-to-end (Perspective Warp -> YOLO Detect -> FEN -> 4-Panel Visualizer)
├── models/                        # Lưu trữ kết quả huấn luyện & trọng số (.pt)
├── runs/
│   └── predict/                   # Thư mục lưu kết quả visualizer suy luận
├── ok.md                          # Hướng dẫn Ultralytics COCO JSON training
└── requirements.txt               # Các thư viện phụ thuộc (ultralytics, opencv-python, pandas, v.v.)
```

---

## 3. Chi tiết Kỹ thuật & Triển khai Các Script

### A. Định nghĩa Custom Dataset & Trainer (`scripts/dataset.py`)
- **Class `ChessREDDataset(YOLODataset)`**:
  - Nhận tham số `json_file` và `split` (`train`, `val`, `test`).
  - Đọc `annotations.json`, tự động lọc danh sách ảnh theo `split` quy định trong `splits` của ChessRED.
  - Sắp xếp và ánh xạ 12 lớp quân cờ sang các ID từ 0 đến 11 (`0: white-pawn`, `1: white-knight`, ..., `11: black-king`).
  - Chuyển đổi bounding box từ COCO pixel `[x, y, w, h]` (Top-Left) sang YOLO normalized center `[cx, cy, w, h]`:
    $$\text{cx} = \frac{x + w/2}{W}, \quad \text{cy} = \frac{y + h/2}{H}, \quad \text{w\_norm} = \frac{w}{W}, \quad \text{h\_norm} = \frac{h}{H}$$
  - Lưu nhãn đã parse vào cache `.cache`.

- **Class `ChessREDTrainer(DetectionTrainer)`**:
  - Ghi đè phương thức `build_dataset(self, img_path, mode="train", batch=None)` để trả về thể hiện của `ChessREDDataset`.

```python
# Custom Dataset and Trainer for direct ChessRED COCO JSON training in Ultralytics YOLO
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import TQDM, colorstr


class ChessREDDataset(YOLODataset):
    """Custom YOLODataset reading ChessRED JSON annotations directly."""

    def __init__(self, *args, json_file="", split="train", **kwargs):
        self.json_file = json_file
        self.split = split
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """Image files are resolved directly from JSON split list."""
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse ChessRED COCO JSON and cache normalized YOLO labels."""
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        # Map categories sorted by ID to 0-indexed sequential classes
        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        # Filter image IDs for the current split
        split_img_ids = set(coco["splits"][self.split]["image_ids"])

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]["pieces"]:
            if ann["image_id"] in split_img_ids:
                img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc=f"Reading {self.split} annotations"):
            if img_info["id"] not in split_img_ids:
                continue

            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["path"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2.0  # top-left [x, y] to center [cx, cy]
                box[[0, 2]] /= w         # normalize x & width
                box[[1, 3]] /= h         # normalize y & height
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
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
        """Load labels from .cache file or parse JSON if cache is missing."""
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
    """Custom DetectionTrainer using ChessREDDataset for in-memory JSON training."""

    def build_dataset(self, img_path, mode="train", batch=None):
        json_file = self.data["annotations_json"]
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
```

---

### B. Cấu hình Data YAML (`datasets/dataset.yaml`)
```yaml
path: datasets/images          # Root directory containing original images
train: train
val: val
test: test

# Path to original ChessRED annotations JSON file
annotations_json: datasets/annotations.json

nc: 12
names:
  0: white-pawn
  1: white-knight
  2: white-bishop
  3: white-rook
  4: white-queen
  5: white-king
  6: black-pawn
  7: black-knight
  8: black-bishop
  9: black-rook
  10: black-queen
  11: black-king
```

---

### C. Script Fine-tune YOLOv11m (`scripts/train.py`)
```python
# Fine-tune YOLOv11m using custom ChessREDTrainer & Draccus
from ultralytics import YOLO
from scripts.dataset import ChessREDTrainer

def run_training():
    model = YOLO("yolo11m.pt")
    results = model.train(
        data="datasets/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="yolo11m_chessred",
        project="models",
        trainer=ChessREDTrainer,
        device=0
    )

if __name__ == "__main__":
    run_training()
```

---

### D. Tiền xử lý biến đổi góc nhìn bàn cờ (`scripts/preprocess_perspective.py`)
Triển khai module nắn góc nhìn bàn cờ từ `Dynamic-Chess-Board-Piece-Extraction`:
1. Chuyển Grayscale -> OTSU threshold -> Canny edge detection -> Dilation.
2. Trích xuất các đường grid bằng `cv2.HoughLinesP`.
3. Phát hiện contours & lọc các ô cờ hợp lệ (`cv2.approxPolyDP`).
4. Xác định 4 góc cực trị bàn cờ $P_\text{top\_left}, P_\text{top\_right}, P_\text{bottom\_left}, P_\text{bottom\_right}$.
5. Thực hiện `cv2.getPerspectiveTransform` và `cv2.warpPerspective` nắn phẳng góc nhìn bàn cờ 2D (Top-down view).
6. Tính toán tọa độ đảo ngược `M_inv` để vẽ 64 ô cờ chính xác lên ảnh gốc ban đầu.

---

### E. Script Inference & Visualizer (`scripts/predict.py`)
Workflow suy luận kết hợp Perspective Transformation & YOLO Detection:
1. **Nhận ảnh nghiêng**: Đọc ảnh bàn cờ đầu vào (ảnh nghiêng gốc).
2. **Nắn phẳng góc nhìn & Trích xuất 64 ô cờ**: Gọi `preprocess_perspective.py` để tìm 4 góc bàn cờ, tính ma trận biến đổi $M$ và $M^{-1}$, trích xuất tọa độ 64 ô cờ.
3. **Phát hiện quân cờ**: Sử dụng mô hình YOLOv11m đã fine-tuned (`best.pt`) để phát hiện quân cờ trên ảnh gốc.
4. **Ánh xạ quân cờ vào 64 ô cờ & Xuất mã FEN**: Ánh xạ tọa độ tâm nêm chân quân cờ $(x_{\text{mid}}, y_{\text{mid}})$ vào 64 đa giác ô cờ để tạo mã FEN chính xác.
5. **Trực quan hóa (Visualizer)**:
   - Tạo ảnh ghép 4 khung hình dạng lưới 2x2 theo đúng trình tự pipeline xử lý (từ trái qua phải, từ trên xuống dưới):
     - **Hàng 1 - Cột 1 (Trái trên)**: `1. Raw Image` (Ảnh nghiêng gốc).
     - **Hàng 1 - Cột 2 (Phải trên)**: `2. Extracted Squares` (Ảnh trích xuất lưới 64 ô cờ sau biến đổi góc nhìn).
     - **Hàng 2 - Cột 1 (Trái dưới)**: `3. YOLO Bounding Box` (Ảnh kết quả phát hiện quân cờ với Bounding Box YOLO).
     - **Hàng 2 - Cột 2 (Phải dưới)**: `4. Converted Image` (Bàn cờ 2D trực quan hóa từ mã FEN).
6. **Lưu kết quả**: Xuất ảnh trực quan 4-panel và mã FEN vào `chess_pieces_detection/runs/predict/`.

---

## 4. Kế hoạch Kiểm thử & Xác minh (Verification Plan)

### A. Kiểm tra Data Parser & Cache
- Chạy thử nghiệm khởi tạo `ChessREDDataset` với `split='val'` để kiểm tra quá trình parse `annotations.json` và tốc độ sinh file `chessred_val.cache`.
- Visual check: Trực quan hóa dữ liệu nhãn thu được từ dataset để đảm bảo bounding box khớp đúng quân cờ.

### B. Kiểm tra Baseline Training
- Chạy 1 epoch thử nghiệm với `yolo11m.pt` và `ChessREDTrainer` để xác nhận pipeline huấn luyện chạy trôi chảy không có lỗi nạp dữ liệu hay bộ nhớ GPU.

### C. Kiểm tra End-to-End Prediction (`predict.py`)
- Truyền ảnh bàn cờ nghiêng từ `datasets/images/` qua `scripts/predict.py`.
- Kiểm tra việc nắn góc nhìn, trích xuất 64 ô cờ, nhận diện quân cờ và lưu kết quả 3 panel trong `runs/predict/`.


# New architecture

Chúng ta có thể hợp nhất phương pháp YOLO Segmentation + Base Point Projection của AI_Chess với cấu trúc Clean Architecture / Python Package của dự án chess_pieces_detection theo thư mục chuẩn sau:

text
chess_pieces_detection/
├── pyproject.toml                  # Đóng gói Python package & dependencies
├── configs/
│   └── predict_config.yaml         # File cấu hình YAML (đường dẫn model, threshold, device)
├── chess_pieces_detection/         # Core Python Package
│   ├── __init__.py
│   ├── core/
│   │   ├── board_segmentor.py      # YOLO Segmentation + RANSAC line fitting tìm 4 góc
│   │   ├── perspective_transformer.py # Biến đổi góc nhìn & trích xuất 64 ô cờ
│   │   └── piece_detector.py       # YOLO Piece Detection & Ánh xạ chân quân cờ vào ô
│   ├── fen/
│   │   ├── fen_builder.py          # Chuyển đổi vị trí 64 ô thành mã FEN chuẩn
│   │   └── stockfish_engine.py     # Tích hợp Stockfish gợi ý nước đi tối ưu
│   └── visualization/
│       ├── board_2d_renderer.py    # Vẽ bàn cờ 2D đồ họa đẹp mắt (Matplotlib / Unicode)
│       └── composite_visualizer.py # Trực quan hóa ảnh ghép 4-Panel 2x2
└── scripts/
    ├── predict.py                  # CLI suy luận End-to-End (đọc config YAML)
    └── train.py                    # Script huấn luyện YOLO


chess_pieces_detection/
├── pyproject.toml                  # Đóng gói Python package & dependencies
├── configs/
│   └── predict_config.yaml         # File cấu hình YAML (đường dẫn model, threshold, device)
├── src/         # Core Python Package
│   ├── __init__.py
│   ├── board_segmentor.py      # YOLO Segmentation + RANSAC line fitting tìm 4 góc
│   ├── perspective_transformer.py # Biến đổi góc nhìn & trích xuất 64 ô cờ
│   └── piece_detector.py       # YOLO Piece Detection & Ánh xạ chân quân cờ vào ô
│   ├── fen_builder.py          # Chuyển đổi vị trí 64 ô thành mã FEN chuẩn
│   ├── result_visualizer.py    # Vẽ bàn cờ 2D đồ họa đẹp mắt (Matplotlib / Unicode) và Trực quan hóa ảnh ghép 4-Panel 2x2
└── scripts/
    ├── predict.py                  # CLI suy luận End-to-End (đọc config YAML)
    └── train.py                    # Script huấn luyện YOLO