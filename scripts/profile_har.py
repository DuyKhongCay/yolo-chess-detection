"""HAR model profiling script using Hailo DFC API and Draccus configuration."""

from dataclasses import dataclass
from pathlib import Path
import draccus
from hailo_sdk_client import ClientRunner


@dataclass
class ProfileHarConfig:
    """Dataclass configuration for HAR model profiling."""

    har: str = "runs/chess_detection_yolo11n/weights/best_hailo_model/best.har"
    target: str = "hailo8"
    output_dir: str = "runs/chess_detection_yolo11n/weights/best_hailo_model/profiler_output"


@draccus.wrap()
def profile_har_model(cfg: ProfileHarConfig):
    """Profile HAR model using Hailo DFC API and export full performance metrics."""
    har_path = Path(cfg.har)
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Profiling HAR model: {har_path} for target architecture: {cfg.target}...")

    runner = ClientRunner(har_path=str(har_path))

    # Run full profiler analysis
    profile_report = runner.profile(target=cfg.target)

    # Export text summary report
    report_txt = output_path / "profiler_summary.txt"
    with open(report_txt, "w") as f:
        f.write(str(profile_report))
    print(f"Profiler summary report saved to: {report_txt}")

    # Generate interactive HTML report
    try:
        html_report_path = output_path / "profiler_report.html"
        runner.generate_profiler_html_report(output_path=str(html_report_path))
        print(f"Interactive HTML profiler report saved to: {html_report_path}")
    except AttributeError:
        print("Note: Profiler report generation completed via runner.profile().")


if __name__ == "__main__":
    profile_har_model()
