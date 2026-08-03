"""Hailo NPU Runtime benchmarking script using HailoRT Python API and Draccus configuration."""

from dataclasses import dataclass
import time
from pathlib import Path
import draccus
import numpy as np
from hailo_platform import (
    HEF,
    Device,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


@dataclass
class HefRtConfig:
    """Dataclass configuration for HEF runtime benchmark."""

    hef: str = "runs/chess_detection_yolo11n/weights/best_hailo_model/best.hef"
    iterations: int = 200
    warmup: int = 20


def get_device_telemetry(device):
    """Query chip temperature and power consumption metrics."""
    telemetry = {}
    try:
        temp_info = device.get_chip_temperature()
        telemetry["ts0_temperature"] = getattr(temp_info, "ts0_temperature", "N/A")
        telemetry["ts1_temperature"] = getattr(temp_info, "ts1_temperature", "N/A")
    except Exception as e:
        telemetry["temperature_error"] = str(e)

    try:
        power_info = device.get_power_measurement()
        telemetry["power_mw"] = getattr(power_info, "power", "N/A")
    except Exception as e:
        telemetry["power_error"] = str(e)

    return telemetry


@draccus.wrap()
def measure_hef_runtime(cfg: HefRtConfig):
    """Run comprehensive performance benchmarking on Hailo NPU hardware."""
    hef_path = Path(cfg.hef)
    print(f"--- Loading HEF Model: {hef_path} ---")
    hef = HEF(str(hef_path))

    # Display HEF stream information
    input_vstream_infos = hef.get_input_vstream_infos()
    output_vstream_infos = hef.get_output_vstream_infos()

    print("\n[Input VStreams]")
    total_input_bytes = 0
    for info in input_vstream_infos:
        print(f"  - Name: {info.name}, Shape: {info.shape}, Format: {info.format.type}")
        total_input_bytes += np.prod(info.shape)

    print("\n[Output VStreams]")
    for info in output_vstream_infos:
        print(f"  - Name: {info.name}, Shape: {info.shape}, Format: {info.format.type}")

    # Configure VDevice and network parameters
    vdevice_params = VDevice.create_params()
    vdevice = VDevice(vdevice_params)

    # Query physical device for telemetry if available
    physical_devices = vdevice.get_physical_devices()
    target_device = physical_devices[0] if len(physical_devices) > 0 else None

    if target_device:
        initial_telemetry = get_device_telemetry(target_device)
        print(f"\n[Initial Telemetry] Temp: {initial_telemetry.get('ts0_temperature', 'N/A')}°C")

    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_group = vdevice.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.AUTO)
    output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.AUTO)

    # Prepare dummy input data with explicit 4D batch dimension (1, H, W, C)
    input_data = {}
    for info in input_vstream_infos:
        input_data[info.name] = np.random.randint(0, 255, size=(1, *info.shape), dtype=np.uint8)

    print(f"\n--- Starting Benchmark: {cfg.warmup} warmup runs + {cfg.iterations} test iterations ---")

    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            # Warmup runs
            for _ in range(cfg.warmup):
                _ = infer_pipeline.infer(input_data)

            # Benchmark runs with individual latency collection
            latencies_ms = []
            start_total_time = time.perf_counter()

            for _ in range(cfg.iterations):
                t0 = time.perf_counter()
                _ = infer_pipeline.infer(input_data)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

            end_total_time = time.perf_counter()

    # Query telemetry after benchmark run
    final_telemetry = get_device_telemetry(target_device) if target_device else {}

    # Calculate performance metrics
    total_duration_sec = end_total_time - start_total_time
    fps = cfg.iterations / total_duration_sec
    avg_latency_ms = float(np.mean(latencies_ms))
    min_latency_ms = float(np.min(latencies_ms))
    max_latency_ms = float(np.max(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    mb_per_sec = (total_input_bytes * cfg.iterations) / (1024 * 1024 * total_duration_sec)

    # Report benchmark results
    print("\n==================================================")
    print("           HAILO NPU RUNTIME BENCHMARK RESULTS     ")
    print("==================================================")
    print(f"Total Iterations    : {cfg.iterations}")
    print(f"Total Execution Time: {total_duration_sec:.4f} sec")
    print(f"Throughput (FPS)    : {fps:.2f} FPS")
    print(f"Avg Latency         : {avg_latency_ms:.2f} ms")
    print(f"Min Latency         : {min_latency_ms:.2f} ms")
    print(f"Max Latency         : {max_latency_ms:.2f} ms")
    print(f"95th Percentile Lat : {p95_latency_ms:.2f} ms")
    print(f"Input Bandwidth     : {mb_per_sec:.2f} MB/s")
    if target_device:
        print(f"Final Chip Temp     : {final_telemetry.get('ts0_temperature', 'N/A')}°C")
        if "power_mw" in final_telemetry and final_telemetry["power_mw"] != "N/A":
            print(f"Power Consumption   : {final_telemetry['power_mw']} mW")
    print("==================================================")


if __name__ == "__main__":
    measure_hef_runtime()
