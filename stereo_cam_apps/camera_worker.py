import time
import cv2
import numpy as np
import yaml
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


@dataclass
class StereoCameraConfig:
    """Dataclass holding stereocamera parameters and UI configurations."""
    sensor_res: List[int] | None = None
    output_res: List[int] | None = None
    window_size: List[int] | None = None
    vflip: bool = False
    rotation: int = 0
    first_run: bool = False
    hdr: str = "off"
    framerate: float = 30.0
    denoise: str = "auto"
    awb: str = "auto"
    ae: bool = True
    ev: float = 0.0

    # Realtime camera controls
    gain: float = 1.0
    exposure_time: int = 20000
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0
    zoom: float = 1.0

    # Tuning file and Save paths
    encoding: str = "jpg"
    tuning_file: str | None = None
    left_save_dir: str | None = None
    right_save_dir: str | None = None
    yaml_path: str = "/home/duykhongcay/hailo_ws/chess_pieces_detection/configs/stereo_camera_config.yaml"

    def __post_init__(self) -> None:
        """Validate configuration parameters and raise ValueError if required items are missing."""
        required_fields = [
            "sensor_res",
            "output_res",
            "left_save_dir",
            "right_save_dir",
            "yaml_path",
        ]
        for field_name in required_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"Missing required configuration parameter: '{field_name}' in config YAML")

    def save_to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        with open(yaml_path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def apply_zoom(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Crop center region of image based on zoom_factor."""
    if zoom_factor <= 1.0:
        return image
    h, w = image.shape[:2]
    crop_w = max(1, int(w / zoom_factor))
    crop_h = max(1, int(h / zoom_factor))
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]


def letterbox_resize(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target_size (width, height) maintaining aspect ratio."""
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded


class DualStereoCamera:
    """Manages dual IMX219 camera instances using Picamera2 or OpenCV fallback."""

    def __init__(self, config: StereoCameraConfig):
        self.config = config
        self.cam_left = None
        self.cam_right = None
        self.use_picam = PICAMERA2_AVAILABLE

        self._init_cameras()

    def _init_cameras(self) -> None:
        """Initialize left and right camera streams with optional tuning_file."""
        if self.use_picam:
            try:
                kw_right = {"camera_num": 0}
                kw_left = {"camera_num": 1}
                if self.config.tuning_file:
                    kw_left["tuning"] = self.config.tuning_file
                    kw_right["tuning"] = self.config.tuning_file

                self.cam_left = Picamera2(**kw_left)
                self.cam_right = Picamera2(**kw_right)

                cam_config_left = self.cam_left.create_still_configuration(
                    main={"size": tuple(self.config.sensor_res)}
                )
                cam_config_right = self.cam_right.create_still_configuration(
                    main={"size": tuple(self.config.sensor_res)}
                )

                self.cam_left.configure(cam_config_left)
                self.cam_right.configure(cam_config_right)

                self.cam_left.start()
                self.cam_right.start()

                self.apply_controls(self.config)
            except Exception as e:
                print(f"[Warning] Failed to initialize Picamera2: {e}. Switching to synthetic fallback.")
                self.use_picam = False

    def read_default_controls(self) -> dict:
        """Read default control values from camera metadata."""
        if not self.use_picam or not self.cam_left:
            return {}
        try:
            meta = self.cam_left.capture_metadata()
            return {
                "gain": float(meta.get("AnalogueGain", 1.0)),
                "exposure_time": int(meta.get("ExposureTime", 20000)),
                "brightness": float(meta.get("Brightness", 0.0)),
                "contrast": float(meta.get("Contrast", 1.0)),
                "saturation": float(meta.get("Saturation", 1.0)),
                "sharpness": float(meta.get("Sharpness", 1.0)),
            }
        except Exception as e:
            print(f"[Warning] Failed to read default camera controls: {e}")
            return {}

    def apply_controls(self, config: StereoCameraConfig) -> None:
        """Apply camera control options dynamically to both cameras."""
        self.config = config
        if not self.use_picam:
            return

        controls = {
            "AeEnable": self.config.ae,
            "AwbEnable": (self.config.awb == "auto" or self.config.awb is True),
            "ExposureValue": float(self.config.ev),
            "Brightness": float(self.config.brightness),
            "Contrast": float(self.config.contrast),
            "Saturation": float(self.config.saturation),
            "Sharpness": float(self.config.sharpness),
        }

        if not self.config.ae:
            controls["AnalogueGain"] = float(self.config.gain)
            controls["ExposureTime"] = int(self.config.exposure_time)

        try:
            if self.cam_left:
                self.cam_left.set_controls(controls)
            if self.cam_right:
                self.cam_right.set_controls(controls)
        except Exception as e:
            print(f"[Error] Failed to set camera controls: {e}")

    def capture_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture left and right frames and resize without aspect ratio distortion."""
        target_out = (self.config.output_res[0], self.config.output_res[1])

        if self.use_picam and self.cam_left and self.cam_right:
            frame_l = self.cam_left.capture_array()
            frame_r = self.cam_right.capture_array()
        else:
            # Synthetic frame generator for testing/demo when cameras are unavailable
            frame_l = np.zeros((self.config.sensor_res[1], self.config.sensor_res[0], 3), dtype=np.uint8)
            frame_r = np.zeros((self.config.sensor_res[1], self.config.sensor_res[0], 3), dtype=np.uint8)
            cv2.putText(frame_l, "LEFT CAMERA (IMX219)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(frame_r, "RIGHT CAMERA (IMX219)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        if tuple(self.config.sensor_res) != target_out:
            frame_l = letterbox_resize(frame_l, target_out)
            frame_r = letterbox_resize(frame_r, target_out)

        if self.config.vflip:
            frame_l = cv2.flip(frame_l, 0)
            frame_r = cv2.flip(frame_r, 0)

        if self.config.rotation == 90:
            frame_l = cv2.rotate(frame_l, cv2.ROTATE_90_CLOCKWISE)
            frame_r = cv2.rotate(frame_r, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation == 180:
            frame_l = cv2.rotate(frame_l, cv2.ROTATE_180)
            frame_r = cv2.rotate(frame_r, cv2.ROTATE_180)
        elif self.config.rotation == 270:
            frame_l = cv2.rotate(frame_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_r = cv2.rotate(frame_r, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.config.zoom > 1.0:
            frame_l = apply_zoom(frame_l, self.config.zoom)
            frame_r = apply_zoom(frame_r, self.config.zoom)

        return frame_l, frame_r

    def stop(self) -> None:
        """Stop camera streams."""
        if self.use_picam:
            if self.cam_left:
                self.cam_left.stop()
            if self.cam_right:
                self.cam_right.stop()
