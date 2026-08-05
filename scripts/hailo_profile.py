"""HAR model profiling script using Hailo DFC API and Draccus configuration."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import draccus
from hailo_sdk_client import ClientRunner


@dataclass
class ProfileHailoConfig:
    """Dataclass configuration for HAR model profiling."""

    har: str = ""
    target: str = "hailo8"
    output_dir: str = ""


@draccus.wrap()
def profile_hailo_model(cfg: ProfileHailoConfig):
    """Profile HAR model using Hailo DFC API and export full performance metrics."""
    har_path = Path(cfg.har).resolve()
    output_path = Path(cfg.output_dir).resolve() if cfg.output_dir else har_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Profiling HAR model: {har_path} for target architecture: {cfg.target}...")

    runner = ClientRunner(hw_arch=cfg.target, har=str(har_path))

    # Run full profiler analysis via Python SDK
    profile_report = runner.profile()

    # Export text summary report
    report_txt = output_path / "profiler_summary.txt"
    with open(report_txt, "w") as f:
        f.write(str(profile_report))
    print(f"Profiler summary report saved to: {report_txt}")

    # Generate interactive HTML report using Hailo profiler CLI
    # (Matches '!hailo profiler {har_path}' in DFC tutorial notebook)
    try:
        cmd = ["hailo", "profiler", str(har_path)]
        print(f"Running Hailo profiler CLI: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(output_path), check=True)
        print(f"Interactive HTML profiler report generated in: {output_path}")
    except Exception as e:
        print(f"Warning: Failed to generate profiler HTML report via CLI: {e}")


if __name__ == "__main__":
    profile_hailo_model()
