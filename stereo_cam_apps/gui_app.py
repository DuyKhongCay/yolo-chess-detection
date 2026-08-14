import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QSlider, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
        QFormLayout, QCheckBox, QComboBox, QLineEdit, QFileDialog, QSizePolicy,
        QMessageBox
    )
    from PySide6.QtCore import QTimer, Qt, Slot
    from PySide6.QtGui import QImage, QPixmap
except ImportError:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QSlider, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
        QFormLayout, QCheckBox, QComboBox, QLineEdit, QFileDialog, QSizePolicy,
        QMessageBox
    )
    from PyQt6.QtCore import QTimer, Qt, Slot
    from PyQt6.QtGui import QImage, QPixmap

import draccus
from camera_worker import StereoCameraConfig, DualStereoCamera, letterbox_resize


class ZoomableImageLabel(QLabel):
    """Custom QLabel supporting interactive mouse wheel zoom and drag panning."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_dragging = False
        self.last_pos = None

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom in and zoom out."""
        angle_delta = event.angleDelta().y()
        if angle_delta > 0:
            self.zoom_factor = min(5.0, self.zoom_factor + 0.2)
        else:
            self.zoom_factor = max(1.0, self.zoom_factor - 0.2)
            if self.zoom_factor == 1.0:
                self.pan_x = 0.0
                self.pan_y = 0.0
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Start drag panning on left click when zoomed in."""
        if event.button() == Qt.MouseButton.LeftButton and self.zoom_factor > 1.0:
            self.is_dragging = True
            self.last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Drag pan across zoomed image area."""
        if self.is_dragging and self.last_pos is not None:
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()

            max_pan = (self.zoom_factor - 1.0) / (2.0 * self.zoom_factor)
            w = max(1, self.width())
            h = max(1, self.height())

            self.pan_x -= delta.x() / w / self.zoom_factor
            self.pan_y -= delta.y() / h / self.zoom_factor

            self.pan_x = max(-max_pan, min(max_pan, self.pan_x))
            self.pan_y = max(-max_pan, min(max_pan, self.pan_y))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Stop drag panning on mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop and zoom frame according to zoom_factor and pan offset."""
        if self.zoom_factor <= 1.0:
            return frame
        h, w = frame.shape[:2]
        crop_w = max(1, int(w / self.zoom_factor))
        crop_h = max(1, int(h / self.zoom_factor))

        center_x = int(w / 2.0 + self.pan_x * w)
        center_y = int(h / 2.0 + self.pan_y * h)

        x1 = max(0, min(w - crop_w, center_x - crop_w // 2))
        y1 = max(0, min(h - crop_h, center_y - crop_h // 2))

        return frame[y1:y1 + crop_h, x1:x1 + crop_w]


class ToastNotification(QWidget):
    """Floating non-blocking toast notification popup at the bottom-right corner."""

    def __init__(self, message: str, parent=None, timeout_ms: int = 3000):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        label = QLabel(message, self)
        label.setStyleSheet("""
            background-color: #2e7d32;
            color: white;
            font-size: 13px;
            font-weight: bold;
            padding: 8px 14px;
            border-radius: 6px;
        """)
        layout.addWidget(label)

        self.adjustSize()

        # Position at bottom-right corner of parent window
        if parent:
            p_rect = parent.rect()
            x = p_rect.width() - self.width() - 20
            y = p_rect.height() - self.height() - 20
            self.move(x, y)

        # Auto dismiss timer
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close)
        self.timer.start(timeout_ms)


class StereoCamMainWindow(QMainWindow):
    """Main Window GUI application for IMX219 stereo camera control."""

    def __init__(self, config: StereoCameraConfig):
        super().__init__()
        self.config = config

        self.camera = DualStereoCamera(self.config)

        if self.config.first_run:
            defaults = self.camera.read_default_controls()
            for k, v in defaults.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self.config.first_run = False
            self.config.save_to_yaml(self.config_path)
        else:
            self.camera.apply_controls(self.config)

        self.fps_last_time = time.perf_counter()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        self.setWindowTitle("Stereo Camera Controller (0.0 FPS)")
        
        # Calculate initial window size (default height = 640) to fit 2 horizontal video feeds + control panel
        out_w, out_h = self.config.output_res if self.config.output_res else (1024, 1024)
        aspect_ratio = out_w / out_h
        win_h = 480
        padding = 30
        target_video_h = win_h - padding
        target_video_w = int(target_video_h * aspect_ratio)
        ctrl_w = 320
        win_w = target_video_w * 2 + ctrl_w + padding
        self.resize(win_w, win_h)

        self._init_ui()

        # Timer for frame update
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(int(1000 / self.config.framerate))

    def _init_ui(self) -> None:
        """Initialize GUI components and layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Video Display Panel (Horizontal Layout with Mouse Zoomable Labels)
        video_layout = QHBoxLayout()
        self.label_left = ZoomableImageLabel("Left Camera Feed", self)
        self.label_right = ZoomableImageLabel("Right Camera Feed", self)
        self.label_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_left.setStyleSheet("border: 1px solid gray; background: #1e1e1e;")
        self.label_right.setStyleSheet("border: 1px solid gray; background: #1e1e1e;")
        self.label_left.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.label_right.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.label_left.setMinimumSize(1, 1)
        self.label_right.setMinimumSize(1, 1)

        video_layout.addWidget(self.label_left)
        video_layout.addWidget(self.label_right)

        # Control Panel Sidebar
        control_group = QGroupBox("Camera Parameters", self)
        control_layout = QFormLayout(control_group)

        # Gain
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(1.0, 16.0)
        self.spin_gain.setSingleStep(0.1)
        self.spin_gain.setValue(self.config.gain)
        self.spin_gain.valueChanged.connect(self.on_config_changed)
        control_layout.addRow("Gain:", self.spin_gain)

        # Exposure Time
        self.spin_exposure = QSpinBox()
        self.spin_exposure.setRange(100, 100000)
        self.spin_exposure.setSingleStep(500)
        self.spin_exposure.setValue(self.config.exposure_time)
        self.spin_exposure.valueChanged.connect(self.on_config_changed)
        control_layout.addRow("Exposure (us):", self.spin_exposure)

        # Brightness
        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.setRange(-1.0, 1.0)
        self.spin_brightness.setSingleStep(0.05)
        self.spin_brightness.setValue(self.config.brightness)
        self.spin_brightness.valueChanged.connect(self.on_config_changed)
        control_layout.addRow("Brightness:", self.spin_brightness)

        # Contrast
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.0, 2.0)
        self.spin_contrast.setSingleStep(0.05)
        self.spin_contrast.setValue(self.config.contrast)
        self.spin_contrast.valueChanged.connect(self.on_config_changed)
        control_layout.addRow("Contrast:", self.spin_contrast)

        # Saturation
        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setRange(0.0, 2.0)
        self.spin_saturation.setSingleStep(0.05)
        self.spin_saturation.setValue(self.config.saturation)
        self.spin_saturation.valueChanged.connect(self.on_config_changed)
        control_layout.addRow("Saturation:", self.spin_saturation)

        # Sharpness
        self.spin_sharpness = QDoubleSpinBox()
        self.spin_sharpness.setRange(0.0, 2.0)
        self.spin_sharpness.setSingleStep(0.05)
        self.spin_sharpness.setValue(self.config.sharpness)
        self.spin_sharpness.valueChanged.connect(self.on_config_changed)
        control_layout.addRow("Sharpness:", self.spin_sharpness)

        # Save Button
        self.btn_capture = QPushButton("Save Stereo Pair", self)
        self.btn_capture.setStyleSheet("background-color: #2b5b84; color: white; font-weight: bold; padding: 8px;")
        self.btn_capture.clicked.connect(self.capture_stereo_images)
        control_layout.addRow(self.btn_capture)

        # Status Label
        self.label_status = QLabel("Ready", self)
        control_layout.addRow(self.label_status)

        control_group.setFixedWidth(320)

        main_layout.addLayout(video_layout, stretch=4)
        main_layout.addWidget(control_group, stretch=1)

    @Slot()
    def on_config_changed(self) -> None:
        """Update config data and persist parameters to YAML config file."""
        self.config.gain = float(self.spin_gain.value())
        self.config.exposure_time = int(self.spin_exposure.value())
        self.config.brightness = float(self.spin_brightness.value())
        self.config.contrast = float(self.spin_contrast.value())
        self.config.saturation = float(self.spin_saturation.value())
        self.config.sharpness = float(self.spin_sharpness.value())

        # Update live camera controls
        self.camera.apply_controls(self.config)

        # Persist to YAML file
        self.config.save_to_yaml(self.config_path)
        self.label_status.setText("Config Saved.")

    def update_frames(self) -> None:
        """Fetch and render camera frames to the UI widgets."""
        frame_l, frame_r = self.camera.capture_frames()

        proc_l = self.label_left.process_frame(frame_l)
        proc_r = self.label_right.process_frame(frame_r)

        pix_l = self._cv_to_pixmap(proc_l)
        pix_r = self._cv_to_pixmap(proc_r)

        self.label_left.setPixmap(pix_l.scaled(
            self.label_left.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        self.label_right.setPixmap(pix_r.scaled(
            self.label_right.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

        # Calculate real-time FPS and update window title bar
        self.fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self.fps_last_time
        if elapsed >= 0.5:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_last_time = now
            self.setWindowTitle(f"Stereo Camera Controller ({self.current_fps:.1f} FPS)")

    def capture_stereo_images(self) -> None:
        """Capture and save 2 separate images from left and right lenses using sequential 4-digit numbering."""
        frame_l, frame_r = self.camera.capture_frames()

        os.makedirs(self.config.left_save_dir, exist_ok=True)
        os.makedirs(self.config.right_save_dir, exist_ok=True)

        next_idx = self._get_next_image_index()
        num_str = f"{next_idx:04d}"

        file_l = os.path.join(self.config.left_save_dir, f"left_{num_str}.jpg")
        file_r = os.path.join(self.config.right_save_dir, f"right_{num_str}.jpg")

        # Convert Picamera2 RGB array to BGR for cv2.imwrite to maintain original color format
        frame_l_bgr = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
        frame_r_bgr = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)

        cv2.imwrite(file_l, frame_l_bgr)
        cv2.imwrite(file_r, frame_r_bgr)

        msg = f"Saved: left_{num_str}.jpg & right_{num_str}.jpg"
        self.label_status.setText(msg)
        print(f"[Captured] Left: {file_l} | Right: {file_r}")

        # Show floating toast notification at bottom-right corner, auto disappearing in 3s
        toast_msg = f"✓ Đã lưu: left_{num_str}.jpg & right_{num_str}.jpg"
        self.toast = ToastNotification(toast_msg, self, timeout_ms=3000)
        self.toast.show()

    def _get_next_image_index(self) -> int:
        """Scan left and right save directories for existing formatted files and return next available 0-indexed integer."""
        max_idx = -1
        for save_dir, prefix in [(self.config.left_save_dir, "left_"), (self.config.right_save_dir, "right_")]:
            if not os.path.exists(save_dir):
                continue
            for fname in os.listdir(save_dir):
                if fname.startswith(prefix) and fname.endswith(".jpg"):
                    num_part = fname[len(prefix):-4]
                    if num_part.isdigit():
                        max_idx = max(max_idx, int(num_part))
        return max_idx + 1

    def _cv_to_pixmap(self, img: np.ndarray) -> QPixmap:
        """Convert an RGB frame to QPixmap."""
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(
            img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        return QPixmap.fromImage(q_img)

    def closeEvent(self, event) -> None:
        """Clean up resources on application close."""
        self.timer.stop()
        self.camera.stop()
        event.accept()


@draccus.wrap(
    config_path="/home/duykhongcay/hailo_ws/chess_pieces_detection/configs/stereo_camera_config.yaml"
)
def main(cfg: StereoCameraConfig) -> None:
    """Main entry point to launch the GUI stereo camera application."""
    app = QApplication(sys.argv)
    window = StereoCamMainWindow(cfg)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
