"""Export YOLO11n PyTorch model to Hailo-8 HEF format using Ultralytics and DFC API."""

from dataclasses import dataclass, field
import os
from pathlib import Path

# Configure TensorFlow environment for GPU execution without pre-initializing CUDA in main process
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_RAM_ALLOCATOR_BYTES_LIMIT"] = "3145728000"  # Limit TF VRAM allocation to ~3000MB for cuDNN workspace
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import draccus
import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.model_translator.exceptions import ParsingWithRecommendationException
    DFC_AVAILABLE = True
except ImportError:
    DFC_AVAILABLE = False


@dataclass
class ConvertModelConfig:
    """Dataclass configuration for YOLO model conversion."""

    model: str = "weights/best.pt"
    dataset: str = "dataset/data.yaml"
    imgsz: int = 640
    method: str = "dfc"  # Choice: "ultralytics" or "dfc"
    name: str = "hailo8"         # Hailo target architecture name
    base_dnn: str = "yolo11n"  # Base DNN model name
    output_dir: str = "output"   # Output directory for DFC export
    calib_count: int = 64       # Calibration dataset size for DFC quantization
    use_memmap: bool = False      # Memory map dataset to SSD disk to support large calibration counts without OOM
    end_node_names: list[str] = field(default_factory=list)  # Custom end node names for ONNX parsing
    optimize_level: int = 1      # Hailo DFC optimization level
    compress_level: int = 0      # Hailo DFC compression level
    model_script: str | None = None  # Custom model script file path or command string for Hailo DFC
    device: str = "cpu"          # Target device for DFC optimization flow ("cpu", "gpu", or GPU index e.g. "0")
    log_dir: str | None = None   # Target directory for Hailo SDK log files (acceleras, allocator, hailort, etc.)


def export_via_ultralytics(model_path: str, dataset_yaml: str, imgsz: int = 640, name: str = "hailo8"):
    """Export model to HEF using Ultralytics built-in Hailo exporter."""
    print(f"[+] Exporting via Ultralytics API for target '{name}'...")
    model = YOLO(model_path)
    export_path = model.export(
        format="hailo",
        name=name,
        imgsz=imgsz,
        data=dataset_yaml
    )
    print(f"[✓] Export completed: {export_path}")
    return export_path


def load_calibration_data(
    dataset_yaml: str,
    imgsz: int = 640,
    count: int = 64,
    use_memmap: bool = True,
    output_dir: str = "output"
):
    """Load calibration images array for Hailo DFC optimization using memory-mapped SSD storage."""
    import yaml
    with open(dataset_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get('path', ''))
    val_sub = Path(data_cfg.get('train', 'images'))
    img_dir = base_path / val_sub if not val_sub.is_absolute() else val_sub

    img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
    img_paths = img_paths[:count]
    if not img_paths:
        raise FileNotFoundError(f"No calibration images found for dataset: {dataset_yaml}")

    total_count = len(img_paths)
    print(f"[+] Loaded {total_count} calibration images across dataset splits ({imgsz}x{imgsz})...")

    if use_memmap:
        cache_path = Path(output_dir) / f"calib_data_{total_count}_{imgsz}x{imgsz}.dat"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        shape = (total_count, imgsz, imgsz, 3)
        expected_size = total_count * imgsz * imgsz * 3 * 4  # float32 bytes

        # Check if pre-cached memmap file already exists on SSD
        if cache_path.exists() and cache_path.stat().st_size == expected_size:
            print(f"[✓] Loading existing SSD memory-mapped dataset from: {cache_path}")
            return np.memmap(cache_path, dtype=np.float32, mode='r', shape=shape)

        print(f"[+] Creating SSD memory-mapped file at: {cache_path} ({expected_size / (1024**3):.2f} GB)...")
        calib_array = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=shape)

        try:
            from tqdm import tqdm
            pbar = tqdm(img_paths, desc="Writing SSD memmap")
        except ImportError:
            pbar = img_paths

        for i, p in enumerate(pbar):
            img = Image.open(p).convert('RGB').resize((imgsz, imgsz))
            calib_array[i] = np.array(img, dtype=np.float32) / 255.0

        calib_array.flush()
        print(f"[✓] SSD memory-mapped calibration data ready ({cache_path.stat().st_size / (1024**3):.2f} GB on SSD)")
        return calib_array
    else:
        calib_images = []
        for p in img_paths:
            img = Image.open(p).convert('RGB').resize((imgsz, imgsz))
            arr = np.array(img, dtype=np.float32) / 255.0
            calib_images.append(arr)

        calib_array = np.array(calib_images, dtype=np.float32)
        print(f"[✓] Calibration data prepared in RAM with shape {calib_array.shape} ({calib_array.nbytes / (1024**2):.1f} MB RAM)")
        return calib_array


def export_via_dfc(
    model_path: str,
    dataset_yaml: str,
    imgsz: int = 640,
    name: str = "hailo8",
    base_dnn: str = "yolo11n_chess",
    output_dir: str = "output",
    calib_count: int = 64,
    use_memmap: bool = True,
    end_node_names: list[str] = None,
    optimize_level: int = 1,
    compress_level: int = 0,
    model_script: str | None = None,
    device: str = "cpu",
    log_dir: str | None = None
):
    """Export model to HEF step-by-step using Hailo DFC Python API."""
    if not DFC_AVAILABLE:
        raise RuntimeError("Hailo Dataflow Compiler (hailo_sdk_client) is not installed in current environment.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure target log directory for Hailo SDK log files and mute screen logs
    target_log_dir = Path(log_dir) if log_dir else out_dir / "logs"
    target_log_dir.mkdir(parents=True, exist_ok=True)
    abs_log_dir = str(target_log_dir.resolve())

    os.environ["HAILO_SDK_LOG_DIR"] = abs_log_dir
    os.environ["HAILORT_LOGGER_PATH"] = str((target_log_dir / "hailort.log").resolve())
    os.environ["HAILO_LOG_DIR"] = abs_log_dir
    os.environ["LOGLEVEL"] = "ERROR"

    import logging
    for lg_name in ["hailo_sdk", "hailo_sdk_client", "hailo_model_optimization", "acceleras", "allocator"]:
        logging.getLogger(lg_name).setLevel(logging.ERROR)

    # Configure TensorFlow / CUDA device execution environment
    if device and device.lower() in ("cpu", "-1"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[+] DFC Optimization device target: CPU")
    elif device and device.lower() in ("gpu", "cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[+] DFC Optimization device target: GPU (cuda:0)")
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        print(f"[+] DFC Optimization device target: GPU (cuda:{device})")

    print("[+] Step 1: Exporting PyTorch model to ONNX...")
    model = YOLO(model_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, device="cpu")

    hef_path = out_dir / f"{base_dnn}_{name}.hef"

    print("[+] Step 2: DFC Parsing ONNX model...")
    runner = ClientRunner(hw_arch=name)
    end_nodes = end_node_names if end_node_names else None

    try:
        runner.translate_onnx_model(onnx_path, net_name=base_dnn, end_node_names=end_nodes)
    except ParsingWithRecommendationException as e:
        if e.recommended_end_node_names:
            print(f"[!] DFC Parsing recommendation triggered. Retrying with recommended end nodes: {e.recommended_end_node_names}...")
            runner = ClientRunner(hw_arch=name)
            runner.translate_onnx_model(onnx_path, net_name=base_dnn, end_node_names=e.recommended_end_node_names)
        else:
            raise

    print("[+] Step 3: Preparing Calibration Data & Optimizing...")
    calib_data = load_calibration_data(
        dataset_yaml,
        imgsz=imgsz,
        count=calib_count,
        use_memmap=use_memmap,
        output_dir=output_dir
    )

    # Load custom model script file or command string if specified
    if model_script:
        try:
            if Path(model_script).is_file():
                print(f"[+] Loading custom model script file: {model_script}")
                runner.load_model_script(model_script)
            else:
                print("[+] Loading custom model script commands...")
                runner.load_model_script(model_script)
        except Exception as e:
            print(f"[!] Warning: Failed to load model_script: {e}")

    # Configure DFC optimization flavor level
    if optimize_level is not None or compress_level is not None:
        opts = []
        if optimize_level is not None:
            opts.append(f"optimization_level={optimize_level}")
        if compress_level is not None:
            opts.append(f"compression_level={compress_level}")
        flavor_cmd = f"model_optimization_flavor({', '.join(opts)})\n"
        try:
            runner.load_model_script(flavor_cmd)
        except Exception as e:
            print(f"[!] Warning: Failed to load model optimization flavor: {e}")

    runner.optimize(calib_data)

    print("[+] Step 4: Compiling Model to HEF...")
    runner.compile()
    hef_data = runner.get_hef()
    # Save HAR model archive for profiler usage
    har_path = out_dir / f"{base_dnn}_{name}.har"
    runner.save_har(har_path)

    with open(hef_path, 'wb') as f:
        f.write(hef_data)

    print(f"[✓] DFC Export completed: {hef_path} and HAR saved to {har_path}")
    return str(hef_path)


@draccus.wrap()
def convert_model(cfg: ConvertModelConfig):
    """Main entrypoint for YOLO model conversion using Draccus configuration."""
    if cfg.method == "ultralytics":
        export_via_ultralytics(cfg.model, cfg.dataset, imgsz=cfg.imgsz, name=cfg.name)
    else:
        export_via_dfc(
            cfg.model,
            cfg.dataset,
            imgsz=cfg.imgsz,
            name=cfg.name,
            base_dnn=cfg.base_dnn,
            output_dir=cfg.output_dir,
            calib_count=cfg.calib_count,
            use_memmap=cfg.use_memmap,
            end_node_names=cfg.end_node_names,
            optimize_level=cfg.optimize_level,
            compress_level=cfg.compress_level,
            model_script=cfg.model_script,
            device=cfg.device,
            log_dir=cfg.log_dir
        )

if __name__ == "__main__":
    convert_model()
