# Hướng dẫn Cấu hình Hyperparameters & Data Augmentation trong Ultralytics YOLO

Tài liệu này tổng hợp và dịch chi tiết toàn bộ các tham số huấn luyện (Train Settings) và tăng cường dữ liệu (Data Augmentation) cho mô hình Ultralytics YOLO từ tài liệu chính thức.

---

## 1. Cấu hình Huấn luyện Mô hình (Train Settings)

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `model` | `str` | `None` | Chỉ định file mô hình để huấn luyện. Nhận đường dẫn tới file trọng số `.pt` pretrained hoặc file cấu hình kiến trúc `.yaml`. |
| `data` | `str` | `None` | Đường dẫn tới file cấu hình tập dữ liệu (ví dụ `dataset.yaml`), chứa khai báo đường dẫn ảnh train/val, số lượng lớp và tên các lớp. |
| `epochs` | `int` | `100` | Tổng số epoch huấn luyện. Mỗi epoch đại diện cho một lượt duyệt qua toàn bộ tập dữ liệu. |
| `time` | `float` | `None` | Thời gian huấn luyện tối đa tính bằng giờ. Nếu được thiết lập, tham số này sẽ ghi đè `epochs` và tự động dừng khi hết giờ. |
| `patience` | `int` | `100` | Số epoch chờ đợi mà các chỉ số validation không cải thiện trước khi tự động dừng sớm (Early Stopping) để chống overfitting. |
| `batch` | `int` / `float` | `16` | Kích thước Batch size. Hỗ trợ 3 chế độ: số nguyên cố định (ví dụ `16`), `-1` (tự động điều chỉnh để dùng 60% VRAM GPU), hoặc tỉ lệ VRAM (ví dụ `0.70`). |
| `imgsz` | `int` | `640` | Kích thước ảnh mục tiêu cho huấn luyện. Ảnh sẽ được resize về hình vuông `imgsz`x`imgsz` (giữ nguyên tỉ lệ khung hình). |
| `save` | `bool` | `True` | Cho phép tự động lưu các file checkpoint và trọng số mô hình tốt nhất (`best.pt`, `last.pt`). |
| `save_period` | `int` | `-1` | Tần suất lưu checkpoint mô hình tính theo số epoch (`-1` là tắt tính năng lưu định kỳ). |
| `cache` | `bool` / `str` | `False` | Nạp trước dữ liệu ảnh vào bộ nhớ RAM (`True`/`ram`) hoặc đĩa (`disk`) để tăng tốc độ huấn luyện. |
| `device` | `int` / `str` / `list` | `None` | Chỉ định thiết bị tính toán: GPU đơn (`device=0`), đa GPU (`device=[0,1]`), CPU (`device=cpu`), MPS Apple (`device=mps`), NPU (`device=npu`), hoặc tự động chọn GPU rảnh (`device=-1`). |
| `workers` | `int` | `8` | Số luồng (worker threads) nạp dữ liệu song song (tính trên mỗi GPU). |
| `project` | `str` | `None` | Tên thư mục dự án dùng để lưu trữ các kết quả và log huấn luyện. |
| `name` | `str` | `None` | Tên của lượt chạy huấn luyện (tạo thư mục con bên trong thư mục `project`). |
| `exist_ok` | `bool` | `False` | Nếu là `True`, cho phép ghi đè lên thư mục `project/name` đã tồn tại thay vì tạo thư mục mới. |
| `save_dir` | `str` | `None` | Ghi đè thư mục lưu trữ chính xác, không tự động tăng số thứ tự thư mục khi chạy lại. |
| `pretrained` | `bool` / `str` | `True` | Xác định việc bắt đầu huấn luyện từ trọng số pretrained hay khởi tạo ngẫu nhiên (`pretrained=False`). |
| `optimizer` | `str` | `'auto'` | Lựa chọn bộ tối ưu hóa: `SGD`, `MuSGD`, `Adam`, `Adamax`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, hoặc `'auto'`. |
| `seed` | `int` | `0` | Khởi tạo giá trị seed ngẫu nhiên nhằm đảm bảo tính tái lập kết quả qua các lần chạy. |
| `deterministic` | `bool` | `True` | Bắt buộc sử dụng các thuật toán định tính (deterministic) để đảm bảo kết quả trùng khớp hoàn toàn. |
| `verbose` | `bool` | `True` | Hiển thị chi tiết tiến trình huấn luyện, thanh progress bar và chỉ số metrics từng epoch ra màn hình. |
| `single_cls` | `bool` | `False` | Coi tất cả các lớp trong tập dữ liệu thành 1 lớp duy nhất (dùng cho bài toán phát hiện đối tượng nhị phân). |
| `classes` | `list[int]` | `None` | Danh sách ID các lớp cụ thể được chọn để huấn luyện (bỏ qua các lớp khác). |
| `rect` | `bool` | `False` | Huấn luyện với chiến lược padding tối thiểu theo hình chữ nhật để tăng tốc độ huấn luyện. |
| `multi_scale` | `float` | `0.0` | Thay đổi ngẫu nhiên kích thước ảnh `imgsz` theo từng batch trong khoảng `+/- multi_scale`. |
| `cos_lr` | `bool` | `False` | Sử dụng bộ điều chỉnh tốc độ học Cosine Learning Rate Scheduler giúp mô hình hội tụ tốt hơn. |
| `close_mosaic` | `int` | `10` | Tắt kỹ thuật tăng cường Mosaic trong N epoch cuối cùng để ổn định mô hình trước khi hoàn tất. |
| `resume` | `bool` | `False` | Tiếp tục quá trình huấn luyện từ file checkpoint đã lưu gần nhất (`last.pt`). |
| `amp` | `bool` | `True` | Bật chế độ huấn luyện độ chính xác hỗn hợp tự động (Automatic Mixed Precision - FP16) giúp tiết kiệm VRAM và tăng tốc. |
| `fraction` | `float` | `1.0` | Tỉ lệ phần trăm của tập dữ liệu được sử dụng để huấn luyện (ví dụ `0.5` là 50% dữ liệu). |
| `profile` | `bool` | `False` | Bật đo đạc hiệu năng tốc độ ONNX / TensorRT trong quá trình huấn luyện. |
| `freeze` | `int` / `list` | `None` | Đóng băng N lớp đầu tiên hoặc danh sách các lớp chỉ định để không cập nhật trọng số (dành cho Transfer Learning). |
| `lr0` | `float` | `0.01` | Tốc độ học ban đầu (Initial Learning Rate). |
| `lrf` | `float` | `0.01` | Tốc độ học tối thiểu ở epoch cuối cùng, tính theo tỉ lệ so với `lr0` (`lr0 * lrf`). |
| `momentum` | `float` | `0.937` | Hệ số Momentum cho bộ tối ưu SGD hoặc Beta1 cho bộ tối ưu Adam. |
| `weight_decay` | `float` | `0.0005` | Hệ số suy giảm trọng số L2 Regularization giúp ngăn ngừa overfitting. |
| `warmup_epochs` | `float` | `3.0` | Số epoch dành cho giai đoạn khởi động (warmup), tăng dần tốc độ học từ nhỏ đến `lr0`. |
| `warmup_momentum` | `float` | `0.8` | Momentum ban đầu trong giai đoạn khởi động warmup. |
| `warmup_bias_lr` | `float` | `0.1` | Tốc độ học dành cho tham số bias trong giai đoạn khởi động warmup. |
| `distill_model` | `str` | `None` | Đường dẫn file trọng số của mô hình giáo viên (Teacher model) cho bài toán Chuyển giao kiến thức (Knowledge Distillation). |
| `dis` | `float` | `6.0` | Trọng số của Distillation Loss đóng góp vào tổng Loss. |
| `box` | `float` | `7.5` | Trọng số của Box Loss (độ chính xác tọa độ Bounding Box) trong hàm mất mát. |
| `cls` | `float` | `0.5` | Trọng số của Classification Loss (độ chính xác phân loại lớp). |
| `cls_pw` | `float` | `0.0` | Trọng số lũy thừa xử lý mất cân bằng lớp (Class Imbalance). |
| `dfl` | `float` | `1.5` | Trọng số của Distribution Focal Loss (DFL) giúp tinh chỉnh mép Bounding Box. |
| `pose` | `float` | `12.0` | Trọng số Pose Loss cho các bài toán ước lượng tư thế Pose Estimation. |
| `kobj` | `float` | `1.0` | Trọng số Keypoint Objectness Loss trong mô hình Pose Estimation. |
| `rle` | `float` | `1.0` | Trọng số Residual Log-Likelihood Estimation Loss trong Pose Estimation. |
| `angle` | `float` | `1.0` | Trọng số Angle Loss cho các mô hình phát hiện hộp xoay (Oriented Bounding Box - OBB). |
| `nbs` | `int` | `64` | Kích thước batch chuẩn hóa (Nominal Batch Size) để tính toán loss. |
| `overlap_mask` | `bool` | `True` | Gộp các mask đối tượng thành 1 mask duy nhất trong bài toán Instance Segmentation. |
| `mask_ratio` | `int` | `4` | Tỉ lệ giảm phân giải của Segmentation Mask trong quá trình huấn luyện. |
| `dropout` | `float` | `0.0` | Tỉ lệ Dropout loại bỏ ngẫu nhiên các node mạng để chống overfitting. |
| `val` | `bool` | `True` | Cho phép tự động đánh giá validation định kỳ sau mỗi epoch huấn luyện. |
| `plots` | `bool` | `True` | Tự động vẽ và lưu các biểu đồ metrics (Loss curves, mAP, Confusion Matrix, prediction samples). |
| `compile` | `bool` / `str` | `False` | Sử dụng PyTorch 2.x `torch.compile` với backend Inductor để tối ưu đồ thị tính toán và tăng tốc. |
| `channels_last` | `bool` | `False` | Sử dụng định dạng bộ nhớ NHWC (Channels Last) tăng tốc cho GPU Tensor Core. |
| `max_det` | `int` | `300` | Số lượng đối tượng tối đa được giữ lại cho mỗi ảnh trong giai đoạn validation. |

---

## 2. Tham số Tăng cường Dữ liệu (Augmentation Settings)

Các kỹ thuật Data Augmentation đóng vai trò cốt lõi trong việc nâng cao độ bền vững và khả năng tổng quát hóa của mô hình YOLO đối với dữ liệu thực tế.

| Tham số | Kiểu dữ liệu | Mặc định | Nhiệm vụ hỗ trợ | Khoảng giá trị | Mô tả chi tiết |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hsv_h` | `float` | `0.015` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Thay đổi sắc độ màu (Hue) ngẫu nhiên của ảnh theo tỉ lệ vòng màu. Giúp mô hình thích ứng với các điều kiện ánh sáng và màu sắc khác nhau. |
| `hsv_s` | `float` | `0.7` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Thay đổi độ bão hòa màu (Saturation), ảnh hưởng đến độ đậm nhạt của màu sắc trong ảnh. |
| `hsv_v` | `float` | `0.4` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Thay đổi giá trị độ sáng (Brightness/Value) của ảnh. |
| `degrees` | `float` | `0.0` | `detect`, `segment`, `pose`, `obb` | `0.0 - 180.0` | Xoay ảnh ngẫu nhiên trong khoảng góc chỉ định (độ), giúp mô hình nhận diện vật thể ở nhiều góc nghiêng. |
| `translate` | `float` | `0.1` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Dịch chuyển ảnh ngẫu nhiên theo chiều ngang và dọc theo tỉ lệ kích thước ảnh. |
| `scale` | `float` | `0.5` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Co giãn phóng to/thu nhỏ ảnh ngẫu nhiên, mô phỏng đối tượng ở các khoảng cách xa gần khác nhau so với camera. |
| `shear` | `float` | `0.0` | `detect`, `segment`, `pose`, `obb` | `-180.0 - +180.0` | Biến dạng xô lệch ảnh theo góc chỉ định, mô phỏng góc nhìn nghiêng của đối tượng. |
| `perspective` | `float` | `0.0` | `detect`, `segment`, `pose`, `obb` | `0.0 - 0.001` | Biến đổi góc nhìn không gian 3D (Perspective Transform) ngẫu nhiên cho ảnh. |
| `flipud` | `float` | `0.0` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Xác suất lật ngược ảnh theo chiều dọc (Trên - Dưới). *(Khuyến nghị đặt 0.0 cho bàn cờ)*. |
| `fliplr` | `float` | `0.5` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Xác suất lật ngược ảnh theo chiều ngang (Trái - Phải). |
| `bgr` | `float` | `0.0` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Xác suất tráo đổi thứ tự kênh màu từ RGB sang BGR. |
| `mosaic` | `float` | `1.0` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Kỹ thuật ghép 4 ảnh huấn luyện ngẫu nhiên thành 1 ảnh phức hợp, giúp mô hình học ngữ cảnh và đối tượng nhỏ cực kỳ hiệu quả. |
| `mixup` | `float` | `0.0` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Trộn lẫn 2 ảnh và nhãn tương ứng với nhau theo tỉ lệ trong suốt, giúp tăng khả năng tổng quát hóa và chống nhiễu. |
| `cutmix` | `float` | `0.0` | `detect`, `segment`, `pose`, `obb` | `0.0 - 1.0` | Cắt một vùng của ảnh này và dán đè lên ảnh khác, mô phỏng các tình huống vật thể bị che khuất một phần (Occlusion). |
| `copy_paste` | `float` | `0.0` | `segment` | `0.0 - 1.0` | Sao chép đối tượng từ ảnh này và dán sang ảnh khác trong bài toán Segmentation. |
| `copy_paste_mode` | `str` | `'flip'` | `segment` | - | Chế độ chiến lược Copy-Paste (`'flip'` hoặc `'mixup'`). |
| `auto_augment` | `str` | `'randaugment'`| `classify` | - | Tự động áp dụng chính sách tăng cường dữ liệu (`'randaugment'`, `'autoaugment'`, hoặc `'augmix'`). |
| `erasing` | `float` | `0.4` | `classify` | `0.0 - 1.0` | Xóa ngẫu nhiên một phần diện tích ảnh trong huấn luyện phân loại. |
| `augmentations` | `list` | `None` | `detect`, `segment`, `pose`, `obb` | - | Cho phép truyền danh sách các phép biến đổi tùy chỉnh từ thư viện Albumentations (Python API). |



# Giải thích quy trình training thông qua log  

Dưới đây là giải thích chi tiết và toàn diện các nội dung thu được từ log huấn luyện của bạn:

---

### 1. Khởi tạo Mô hình & Phần cứng (Environment & Model Setup)
* **Thiết bị**: Chạy trên **NVIDIA GeForce RTX 3050 Ti Laptop GPU** với tổng cộng **3.67 GiB VRAM**.
* **Kiến trúc mô hình**: **YOLO11m** (bản Medium) gồm **232 layers**, **20,062,260 tham số** (~20M params) và độ phức tạp tính toán là **68.2 GFLOPs**.
* **Cấu hình lớp (Classes)**: Đã ghi đè (override) số lượng class từ 80 (của tập dữ liệu COCO mặc định) xuống **12 class** (tương ứng 12 loại quân cờ Đen/Trắng).
* **Transfer Learning**: Chuyển giao thành công **643/649** bộ trọng số từ weights tiễn nạp sẵn (`yolo11m.pt`).
* **Kỹ thuật AMP**: Đã kích hoạt **Automatic Mixed Precision (AMP)** giúp tăng tốc độ tính toán và tiết kiệm VRAM.

---

### 2. Xử lý Dữ liệu & Tự động chọn Batch Size (Data & AutoBatch)
* **Dataset Cache**: Đã đọc và tạo file cache cho 10,800 nhãn ảnh:
  * `chessred_train.cache`
  * `chessred_val.cache`
* **AutoBatch (Tự chọn Batch Size)**:
  * Do đặt `batch: -1`, YOLO tự kiểm tra VRAM ở mức giới hạn 60%.
  * Thử `batch=2` tốn 3.58 GB ➔ bị tràn VRAM (CUDA Out of Memory).
  * Do đó, hệ thống chốt sử dụng **`batch-size = 1`** (chiếm ~2.14 GB / 3.67 GB VRAM).

---

### 3. Quá trình Huấn luyện (Training Progress - 1 Epoch)
* **Optimizer**: Sử dụng **AdamW** với Learning Rate khởi tạo là `0.001` và momentum `0.937`.
* **Thời gian huấn luyện**: Hoàn thành 1 epoch trong **7 phút 12 giây** (tốc độ ~15.0 iteration/giây trên 6,479 bước).
* **VRAM thực tế sử dụng**: **0.98 GB** trong suốt quá trình forward/backward pass.
* **Các chỉ số Loss cuối Epoch 1**:
  * `box_loss` (hàm mất mát tọa độ khung hình): **0.6446** *(Khá tốt)*
  * `cls_loss` (hàm mất mát phân loại quân cờ): **3.9000** *(Còn cao vì mới train 1 epoch)*
  * `dfl_loss` (Distribution Focal Loss): **0.5044**

---

### 4. Kết quả Đánh giá trên tập Validation (Validation Metrics)
Tập validation gồm **2,192 ảnh** với **6,132 đối tượng quân cờ**:

| Chỉ số | Giá trị | Giải thích |
| :--- | :--- | :--- |
| **Precision (P)** | **0.0564 (5.64%)** | Tỷ lệ dự đoán đúng trong số các quân cờ được phát hiện (còn nhiều dự đoán sai). |
| **Recall (R)** | **0.4910 (49.1%)** | Mô hình đã bắt đầu nhận diện được khoảng 49.1% tổng số quân cờ thực tế. |
| **mAP@50** | **0.0586 (5.86%)** | Độ chính xác trung bình ở ngưỡng IoU = 0.5. |
| **mAP@50-95** | **0.0398 (3.98%)** | Độ chính xác trung bình ở các ngưỡng IoU từ 0.5 đến 0.95. |

#### Chi tiết theo từng lớp quân cờ:
* Một số lớp như **Tượng đen (`black-bishop`)**, **Vua đen (`black-king`)**, **Xe đen (`black-rook`)**, **Tượng trắng (`white-bishop`)** có **Recall rất cao (> 87% - 98%)**, chứng tỏ mô hình bắt được vị trí quân cờ khá tốt.
* Tuy nhiên, các lớp như **Mã trắng (`white-knight`)**, **Tốt trắng (`white-pawn`)**, **Hậu đen (`black-queen`)** có chỉ số bằng 0 do chưa đủ epoch để mô hình phân biệt rõ các lớp này.

#### Tốc độ suy luận (Inference Speed):
* **Preprocess**: 0.3 ms/ảnh
* **Inference**: 14.5 ms/ảnh (tương đương ~69 FPS)
* **Postprocess**: 0.9 ms/ảnh

---

### 5. Kết xuất & Lưu Checkpoint
* Đã lược bỏ Optimizer trạng thái để nén dung lượng weights từ vài trăm MB xuống **40.5 MB**.
* Đã lưu 2 bản checkpoint:
  * `best.pt`: Bản có mAP tốt nhất.
  * `last.pt`: Bản ở epoch cuối cùng.
* Đường dẫn lưu: `runs/detect/models/yolo11m_chessred-5/weights/`

---

### 📌 Kết luận & Đánh giá chung
1. **Lần chạy thử nghiệm thành công**: Pipeline huấn luyện end-to-end từ đọc JSON, parse nhãn, cache, autobatch đến val đều hoạt động hoàn hảo mà không gặp lỗi.
2. **mAP hiện tại thấp là bình thường**: Do bạn mới cài đặt `epochs: 1` trong file config để test code. Mô hình mới chỉ học được bước đầu (Recall ~49%).
3. **Khuyến nghị**: Để mô hình đạt độ chính xác cao (mAP > 80-90%), bạn nên sửa `epochs: 30` hoặc `epochs: 50` trong file [`configs/train_config.yaml`](file:///home/duykhongcay/lerobot_ws/chess_pieces_detection/configs/train_config.yaml) và chạy lại.