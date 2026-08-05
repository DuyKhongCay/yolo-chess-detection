"""Optimize ONNX model and compile to Hailo HEF using DFC SDK."""

import os
import gc
from pathlib import Path
from dataclasses import dataclass, field
import draccus
import numpy as np
from PIL import Image

# Set environment variables for TensorFlow and Hailo SDK
if "USER" not in os.environ:
    os.environ["USER"] = "root"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_RAM_ALLOCATOR_BYTES_LIMIT"] = "3221225472"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_CUDNN_FRONTEND"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_cudnn_frontend=false"

try:
    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.model_translator.exceptions import ParsingWithRecommendationException
    DFC_AVAILABLE = True
except ImportError:
    DFC_AVAILABLE = False


@dataclass
class OptimizeDFCConfig:
    """Configuration for Hailo DFC model optimization and compilation."""

    onnx_path: str = "output/model.onnx"
    dataset: str = "dataset/data.yaml"
    imgsz: int = 640
    name: str = "hailo8"
    base_dnn: str = "yolo11n"
    output_dir: str = "output"
    calib_count: int = 64
    use_memmap: bool = True
    end_node_names: list[str] = field(default_factory=list)
    model_script: str | None = None
    device: str = "0"


def load_calibration_data(
    dataset_yaml: str,
    imgsz: int = 640,
    count: int = 64,
    use_memmap: bool = True,
    output_dir: str = "output"
) -> np.ndarray:
    """Load and preprocess calibration images for Hailo DFC optimization."""
    import yaml
    with open(dataset_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get('path', ''))
    val_sub = Path(data_cfg.get('train', 'images'))
    img_dir = base_path / val_sub if not val_sub.is_absolute() else val_sub

    img_paths = (list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg')))[:count]
    if not img_paths:
        raise FileNotFoundError(f"No calibration images found for dataset: {dataset_yaml}")

    total_count = len(img_paths)
    print(f"[+] Loaded {total_count} calibration images ({imgsz}x{imgsz})...")

    if use_memmap:
        cache_path = Path(output_dir) / f"calib_data_{total_count}_{imgsz}x{imgsz}.dat"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        shape = (total_count, imgsz, imgsz, 3)
        expected_size = total_count * imgsz * imgsz * 3 * 4

        if cache_path.exists() and cache_path.stat().st_size == expected_size:
            print(f"[✓] Loading existing SSD memory-mapped dataset: {cache_path}")
            return np.memmap(cache_path, dtype=np.float32, mode='r', shape=shape)

        print(f"[+] Creating SSD memory-mapped dataset at: {cache_path}")
        calib_array = np.memmap(cache_path, dtype=np.float32, mode='w+', shape=shape)

        for i, p in enumerate(img_paths):
            img = Image.open(p).convert('RGB').resize((imgsz, imgsz))
            calib_array[i] = np.array(img, dtype=np.float32) / 255.0

        calib_array.flush()
        print(f"[✓] SSD memory-mapped calibration data ready ({cache_path.stat().st_size / (1024**3):.2f} GB)")
        return calib_array

    calib_images = [np.array(Image.open(p).convert('RGB').resize((imgsz, imgsz)), dtype=np.float32) / 255.0 for p in img_paths]
    calib_array = np.array(calib_images, dtype=np.float32)
    print(f"[✓] Calibration data prepared in RAM ({calib_array.shape})")
    return calib_array


def optimize_dfc(cfg: OptimizeDFCConfig) -> str:
    """Optimize ONNX model and compile to HEF format using Hailo DFC."""
    if not DFC_AVAILABLE:
        raise RuntimeError("Hailo Dataflow Compiler (hailo_sdk_client) is not installed.")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.device and cfg.device.lower() in ("cpu", "-1"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[+] DFC Target Device: CPU")
    else:
        dev_idx = "0" if cfg.device.lower() in ("gpu", "cuda") else str(cfg.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = dev_idx
        print(f"[+] DFC Target Device: GPU (cuda:{dev_idx})")

    print(f"[+] Parsing ONNX model: {cfg.onnx_path}")
    runner = ClientRunner(hw_arch=cfg.name)
    end_nodes = cfg.end_node_names if cfg.end_node_names else None

    try:
        runner.translate_onnx_model(cfg.onnx_path, net_name=cfg.base_dnn, end_node_names=end_nodes)
    except ParsingWithRecommendationException as e:
        if e.recommended_end_node_names:
            print(f"[!] Recommendation triggered. Retrying with end nodes: {e.recommended_end_node_names}")
            runner = ClientRunner(hw_arch=cfg.name)
            runner.translate_onnx_model(cfg.onnx_path, net_name=cfg.base_dnn, end_node_names=e.recommended_end_node_names)
        else:
            raise

    print("[+] Loading Calibration Data & Optimizing...")
    calib_data = load_calibration_data(
        cfg.dataset,
        imgsz=cfg.imgsz,
        count=cfg.calib_count,
        use_memmap=cfg.use_memmap,
        output_dir=cfg.output_dir
    )

    if cfg.model_script:
        # load_model_script() expects a file path, not raw string content.
        # Write the script content to a temp .alls file if it is not an existing file path.
        script_path = Path(cfg.model_script)
        if not script_path.exists():
            script_path = out_dir / "model_script.alls"
            script_path.write_text(cfg.model_script)
            print(f"[+] Model script written to: {script_path}")
        runner.load_model_script(str(script_path))

    print("[+] Running DFC Optimization on GPU...")
    runner.optimize(calib_data)

    hef_path = out_dir / f"{cfg.base_dnn}_{cfg.name}.hef"
    har_path = out_dir / f"{cfg.base_dnn}_{cfg.name}.har"

    # Save HAR (optimizer state) before compile to preserve quantization results
    print("[+] Saving optimized HAR archive...")
    try:
        runner.save_har(str(har_path))
        print(f"[✓] HAR saved: {har_path}")
    except Exception as e:
        print(f"[!] Warning: Could not save HAR: {e}")

    print("[+] Compiling model to HEF...")
    # compile() returns HEF bytes directly in this SDK version
    hef_data = runner.compile()
    if not hef_data:
        raise RuntimeError(
            "runner.compile() returned empty/None. "
            "Check DFC logs above for internal errors."
        )
    with open(hef_path, 'wb') as f:
        f.write(hef_data)
    print(f"[✓] HEF written: {hef_path} ({hef_path.stat().st_size / 1024:.1f} KB)")

    print(f"[✓] DFC Optimization completed: {hef_path} (HAR: {har_path})")
    return str(hef_path)


@draccus.wrap()
def main(cfg: OptimizeDFCConfig):
    """Main entrypoint for DFC optimization script."""
    optimize_dfc(cfg)


if __name__ == "__main__":
    main()
