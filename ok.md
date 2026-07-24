Searched for "y_mid"
Searched for "y_mid"
Listed directory cpp-version
Listed directory src
Searched for "box"
Viewed fen_board_generator.cpp:1-261

Thắc mắc của bạn rất chính xác và đây là một bài toán kinh điển trong Computer Vision gọi là **Hiện tượng thị sai do chiều cao vật thể (Parallax / Perspective Height Offset)** khi camera chụp nghiêng.

### 📐 Bản chất của vấn đề
1. **Bàn cờ là mặt phẳng 2D**, nhưng **quân cờ là vật thể 3D có chiều cao** (quân Vua, Hậu, Xe rất cao).
2. Khi camera chụp nghiêng, phần đầu quân cờ sẽ bị chiếu vươn lên phía trên bức ảnh (vùng pixel có tọa độ $y$ nhỏ hơn).
3. Khung Bounding Box do YOLO bao quanh quân cờ sẽ kéo dài từ **đầu quân cờ** tới **chân đế**. Nếu lấy **tâm hình học chuẩn** của Bounding Box:
   $$\text{y\_center} = \frac{y1 + y2}{2}$$
   thì điểm tâm này sẽ bị lệch hẳn lên phía trên và có thể rơi sang **ô cờ phía sau** quân cờ đó.

---

### 🛠 Project này CÓ xử lý trường hợp này không và xử lý như thế nào?

Tác giả của project **CÓ nhận ra vấn đề này** và xử lý bằng một phương pháp **Heuristic Offset (Cộng bù tọa độ thủ công)** trực tiếp trong code.

#### 1. Xử lý bằng `y_mid` Offset (Cộng thêm padding trục Y)
Nếu soi trực tiếp vào file `perspective_transformation.py` (dòng 440-441) hoặc `square_filling.py` (dòng 398-399):

```python
# Find middle of bounding boxes for x and y 
x_mid = int((x1 + x2) / 2) 

# Add padding to y values to push reference point down toward the base
y_mid = int((y1 + y2) / 2) + 25  # Offset by +25 pixels downward
```

* **Cơ chế:** Thay vì lấy đúng tâm `(y1 + y2) / 2`, tác giả **cố tình cộng thêm `+ 25` pixel** (trong notebook phiên bản OAK-D camera là `+ 40` pixel).
* **Tác dụng:** Hành động `+ 25` này sẽ dịch chuyển điểm kiểm tra từ tâm Bounding Box **dịch xuống phía dưới**, tiệm cận về phía **chân đế quân cờ** – nơi quân cờ thực sự tiếp xúc với mặt bàn cờ!

---

### ⚠️ Đánh giá giải pháp của Project

* **Ưu điểm:** Cực kỳ đơn giản, không tốn chi phí tính toán.
* **Nhược điểm:** Việc gán cứng con số `+25` hay `+40` pixel là giải pháp "mẹo" (heuristic). 
  * Quân Vua/Hậu cao hơn quân Chốt rất nhiều $\rightarrow$ mức độ lệch tâm khác nhau.
  * Góc nghiêng camera thay đổi $\rightarrow$ mức độ lệch tâm cũng thay đổi.

---

### 💡 Các giải pháp triệt để hơn trong thực tế (Best Practices)

Nếu bạn muốn phát triển hoặc cải tiến project này để chính xác 100% ở mọi góc nghiêng, các giải pháp chuẩn trong kỹ thuật gồm:

1. **Dùng điểm Mép Dưới Bounding Box (Footprint Point) thay vì điểm Tâm:**
   Chân đế quân cờ luôn nằm sát mép dưới của Bounding Box ($y2$). Vì vậy điểm đại diện nên được tính là:
   $$\text{x\_foot} = \frac{x1 + x2}{2}, \quad \text{y\_foot} = y2 - \epsilon \quad (\text{với } \epsilon \text{ rất nhỏ})$$
   Điểm này luôn nằm trên mặt bàn cờ bất kể quân cờ cao hay thấp.

2. **Chuyển điểm kiểm tra sang Không gian 2D Phẳng (Warped Space):**
   Thay vì kiểm tra trên ảnh gốc bị nghiêng, ta dùng Ma trận Perspective $M$ để **chiếu điểm chân quân cờ $\text{(x\_foot, y\_foot)}$ sang mặt phẳng 2D vuông góc** $1200 \times 1200$px, sau đó mới chia tọa độ cho kích thước ô vuông ($150 \times 150$px). Việc này loại bỏ hoàn toàn góc nghiêng của camera.

3. **Keypoint Detection (Phát hiện điểm chân đế):**
   Train mô hình (như Pose Estimation / YOLOv8-pose) để dự đoán trực tiếp 2 điểm: **Đầu quân cờ** và **Chân đế quân cờ**. Ta chỉ lấy tọa độ điểm Chân đế để map vào ô cờ.