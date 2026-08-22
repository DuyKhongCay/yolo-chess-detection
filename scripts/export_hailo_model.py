"""Export YOLO (.pt / .onnx) model to Hailo-8 HEF format using Hailo DFC ClientRunner API and Draccus."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
import yaml
import numpy as np
from PIL import Image
import draccus

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.sdk_backend.sdk_backend import SDKPaths
    # Monkey-patch is_release property for DFC SDK version checking if needed
    SDKPaths.is_release = property(lambda self: True)
    DFC_AVAILABLE = True
except ImportError:
    DFC_AVAILABLE = False


@dataclass
class ExportHailoConfig:
    """Dataclass configuration for Hailo model export flow."""

    model_name: str                             # Model architecture identifier
    calib_dir: str                              # Path to calibration images directory
    output_dir: str                             # Output directory for .har and .hef files
    model_path: str = ""                        # Input .pt or .onnx model path (configured via config file or CLI)
    hw_arch: str = "hailo8"                      # Hailo target architecture ('hailo8', 'hailo8l', etc.)
    imgsz: int = 640                             # Input image size (square 640x640)
    calib_count: int = 64                        # Number of calibration images to load
    device: str = "cpu"                          # Execution device ('gpu', '0', 'cpu', '-1')
    stereo_model: bool = False                   # Flag indicating dual-input stereo model (requires left/right calib folders)
    end_node_names: list[str] | None = None      # Custom end node names to trim ONNX graph
    model_script: list[str] | None = None        # Custom model script command strings or file path
    stage: str = "all"                           # Pipeline starting stage: 'all', 'optimize', 'quantize', 'compile'

    def __post_init__(self):
        """Validate configuration parameters immediately upon dataclass initialization."""
        if not self.model_name:
            raise ValueError("model_name must be specified.")
        if not self.output_dir:
            raise ValueError("output_dir must be specified.")

        self.stage = str(self.stage).lower().strip()
        valid_stages = ("all", "optimize", "quantize", "compile")
        if self.stage not in valid_stages:
            raise ValueError(f"Invalid stage '{self.stage}'. Valid options are: {valid_stages}")

        if self.stage in ("all", "optimize", "quantize") and not self.calib_dir:
            raise ValueError(f"calib_dir is required when stage='{self.stage}'.")

        if self.model_script is not None and not isinstance(self.model_script, list):
            raise TypeError(f"model_script must be a list of strings (list[str]), got {type(self.model_script).__name__}")

        if self.stereo_model and self.calib_dir:
            left_dir = Path(self.calib_dir) / "left"
            right_dir = Path(self.calib_dir) / "right"
            if not left_dir.is_dir() or not right_dir.is_dir():
                raise FileNotFoundError(
                    f"stereo_model=True requires both 'left/' and 'right/' subdirectories inside "
                    f"calib_dir: '{Path(self.calib_dir).resolve()}'"
                )


def configure_device(device: str):
    """Set device configuration for TensorFlow and PyTorch."""
    dev = str(device).lower().strip()
    if dev in ("cpu", "-1"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("[+] Device target: CPU")
    elif dev in ("gpu", "cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[+] Device target: GPU (cuda:0)")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = dev
        print(f"[+] Device target: GPU (cuda:{dev})")


def load_calibration_dataset(
    calib_dir_path: str | Path,
    count: int = 1024,
    imgsz: int = 640,
    stereo_model: bool = False
) -> np.ndarray | dict[str, np.ndarray]:
    """Load calibration images. Stereo path validation is guaranteed by ExportHailoConfig.__post_init__."""
    img_dir = Path(calib_dir_path)

    # Helper function to read image list
    def _read_images(folder: Path) -> list[Path]:
        exts = ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG")
        paths = []
        for ext in exts:
            paths.extend(folder.glob(ext))
        return sorted(paths)[:count]

    # Helper function to load images into numpy array
    def _load_numpy(paths: list[Path]) -> np.ndarray:
        total = len(paths)
        data = np.zeros((total, imgsz, imgsz, 3), dtype=np.float32)
        for idx, p in enumerate(paths):
            with Image.open(p) as img:
                data[idx] = np.array(img.convert("RGB").resize((imgsz, imgsz)), dtype=np.float32)
        return data

    if stereo_model:
        left_paths = _read_images(img_dir / "left")
        right_paths = _read_images(img_dir / "right")
        num_pairs = min(len(left_paths), len(right_paths))
        print(f"[+] Loading {num_pairs} stereo calibration pairs from: {img_dir}")
        return {"left": _load_numpy(left_paths), "right": _load_numpy(right_paths)}

    paths = _read_images(img_dir)
    if not paths:
        raise FileNotFoundError(f"No calibration images found in: {img_dir}")
    print(f"[+] Loading {len(paths)} calibration images from: {img_dir}")
    return _load_numpy(paths)


def export_hailo_model(cfg: ExportHailoConfig):
    """Run Hailo model export pipeline starting from specified stage."""
    if not DFC_AVAILABLE:
        print("[!] Error: hailo_sdk_client is not installed in current Python environment.", file=sys.stderr)
        sys.exit(1)

    configure_device(cfg.device)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage = cfg.stage

    # Define standardized HAR and HEF file paths matching notebook convention
    native_har_path = out_dir / f"{cfg.model_name}_hailo_model.har"
    optimized_har_path = out_dir / f"{cfg.model_name}_optimized_model.har"
    quantized_har_path = out_dir / f"{cfg.model_name}_quantized_model.har"
    compiled_har_path = out_dir / f"{cfg.model_name}_compiled_model.har"
    hef_path = out_dir / f"{cfg.model_name}.hef"

    print("=" * 70)
    print(f" Hailo Model Export Pipeline - Target Hardware: {cfg.hw_arch} | Stage: '{stage}'")
    print("=" * 70)
    print(f" Output Directory: {out_dir.resolve()}")

    runner = None

    # STAGE: compile (Re-uses existing quantized_har_path)
    if stage == "compile":
        assert os.path.isfile(quantized_har_path), "Please provide valid path for quantized HAR file"
        print(f"[+] Loading existing Quantized HAR: {quantized_har_path}")
        runner = ClientRunner(har=str(quantized_har_path))
        if cfg.model_script:
            script_cmds = "".join(cfg.model_script) if any("\n" in s for s in cfg.model_script) else "\n".join(cfg.model_script) + "\n"
            print("[+] Loading model script commands from config list...")
            runner.load_model_script(script_cmds)
            
    # STAGE: quantize (Re-uses existing optimized_har_path)
    elif stage == "quantize":
        assert os.path.isfile(optimized_har_path), "Please provide valid path for optimized HAR file"
        print(f"[+] Loading existing FP32 Optimized HAR: {optimized_har_path}")
        runner = ClientRunner(har=str(optimized_har_path))
        if cfg.model_script:
            script_cmds = "".join(cfg.model_script) if any("\n" in s for s in cfg.model_script) else "\n".join(cfg.model_script) + "\n"
            print("[+] Loading model script commands from config list...")
            runner.load_model_script(script_cmds)

    # STAGE: optimize (Re-uses existing native_har_path)
    elif stage == "optimize":
        assert os.path.isfile(native_har_path), "Please provide valid path for HAR file"
        print(f"[+] Loading existing Native HAR: {native_har_path}")
        runner = ClientRunner(har=str(native_har_path))
        if cfg.model_script:
            script_cmds = "".join(cfg.model_script) if any("\n" in s for s in cfg.model_script) else "\n".join(cfg.model_script) + "\n"
            print("[+] Loading model script commands from config list...")
            runner.load_model_script(script_cmds)

    # STAGE: all (ONNX translation & Native HAR generation or existing Native HAR load)
    else:  # stage == "all"
        if cfg.model_path and os.path.isfile(cfg.model_path):
            input_path = Path(cfg.model_path)

            # Step 0: Ensure ONNX file exists (Convert PyTorch .pt to .onnx if needed)
            if input_path.suffix.lower() == ".pt":
                if not ULTRALYTICS_AVAILABLE:
                    raise ImportError("ultralytics package is required to convert .pt model to .onnx format.")
                print(f"[+] Step 0: Exporting PyTorch model '{input_path}' to ONNX...")
                yolo_model = YOLO(str(input_path))
                exported_onnx = yolo_model.export(
                    format="onnx",
                    imgsz=cfg.imgsz,
                    simplify=True,
                    device="cpu" if cfg.device.lower() in ("cpu", "-1") else 0
                )
                onnx_path = str(exported_onnx)
            else:
                onnx_path = str(input_path)

            # Step 1: Parse ONNX Model to Native HAR
            print("[+] Step 1: Parsing ONNX model to Native HAR...")
            runner = ClientRunner(hw_arch=cfg.hw_arch)
            end_nodes = cfg.end_node_names if cfg.end_node_names else None

            runner.translate_onnx_model(
                model=onnx_path,
                net_name=cfg.model_name,
                end_node_names=end_nodes,
            )

            if cfg.model_script:
                script_cmds = "".join(cfg.model_script) if any("\n" in s for s in cfg.model_script) else "\n".join(cfg.model_script) + "\n"
                print("[+] Loading model script commands from config list...")
                runner.load_model_script(script_cmds)

            runner.save_har(str(native_har_path))
            print(f"[✓] Saved Native HAR to: {native_har_path}")

        elif os.path.isfile(native_har_path):
            print(f"[+] Loading existing Native HAR: {native_har_path}")
            runner = ClientRunner(har=str(native_har_path))
            if cfg.model_script:
                script_cmds = "".join(cfg.model_script) if any("\n" in s for s in cfg.model_script) else "\n".join(cfg.model_script) + "\n"
                print("[+] Loading model script commands from config list...")
                runner.load_model_script(script_cmds)

        else:
            raise FileNotFoundError(
                f"Neither model_path ('{cfg.model_path}') nor native HAR file ('{native_har_path}') was found for stage '{stage}'."
            )

    # Flow continuation: Run Step 2 (Full-Precision Optimization) if needed
    if stage in ("all", "optimize"):
        print("[+] Step 2: Optimizing Full-Precision Model...")
        runner.optimize_full_precision()
        runner.save_har(str(optimized_har_path))
        print(f"[✓] Saved Optimized HAR to: {optimized_har_path}")

    # Flow continuation: Run Step 3 (INT8 Quantization Optimization) if needed
    if stage in ("all", "optimize", "quantize"):
        print("[+] Step 3: Running INT8 Quantization Optimization...")
        calib_data = load_calibration_dataset(
            cfg.calib_dir,
            count=cfg.calib_count,
            imgsz=cfg.imgsz,
            stereo_model=cfg.stereo_model
        )

        # Inspect HN model input layer names
        hn_layers = runner.get_hn_dict().get("layers", {})
        input_layers = [l_name for l_name, l_info in hn_layers.items() if l_info.get("type") == "input_layer"]

        if cfg.stereo_model:
            # Map left/right arrays to the two HN input layers by their list order
            calib_dataset_dict = {
                input_layers[0]: calib_data["left"],
                input_layers[1]: calib_data["right"],
            }
            print(f"[+] Stereo calibration input mapping: {list(calib_dataset_dict.keys())}")
        else:
            calib_dataset_dict = {l_name: calib_data for l_name in input_layers}
            print(f"[+] Quantization calibration input layers ({len(input_layers)}): {input_layers}")

        runner.optimize(calib_dataset_dict)
        runner.save_har(str(quantized_har_path))
        print(f"[✓] Saved Quantized HAR to: {quantized_har_path}")

    # Flow continuation: Run Step 4 (Compile Quantized HAR File to HEF)
    print("[+] Step 4: Compiling Quantized HAR File to HEF & Saved Compiled HAR...")
    if cfg.model_script:
        script_cmds = "".join(cfg.model_script) if any("\n" in s for s in cfg.model_script) else "\n".join(cfg.model_script) + "\n"
        print("[+] Loading model script commands before compilation...")
        runner.load_model_script(script_cmds)

    hef_bytes = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef_bytes)

    runner.save_har(str(compiled_har_path))
    print(f"[✓] Compiled HEF saved to: {hef_path}")
    print(f"[✓] Compiled HAR saved to: {compiled_har_path}")
    return str(hef_path)


@draccus.wrap(config_path="/home/duykhongcay/hailo_ws/chess_pieces_detection/configs/export_hailo_config.yaml")
def main(cfg: ExportHailoConfig):
    """Main CLI entrypoint for Hailo model export."""
    export_hailo_model(cfg)


if __name__ == "__main__":
    main()
