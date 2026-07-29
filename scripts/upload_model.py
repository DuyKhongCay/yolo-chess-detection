"""Script to deploy/upload trained YOLO model weights to Roboflow."""

import os
from dataclasses import dataclass
from pathlib import Path
import draccus
from roboflow import Roboflow


@dataclass
class UploadModelConfig:
    """Dataclass configuration for Roboflow model upload."""

    # Roboflow API key (can also be read from ROBOFLOW_API_KEY env var)
    api_key: str | None = None
    # Roboflow workspace ID (optional)
    workspace: str | None = None
    # Roboflow project ID
    project: str = ""
    # Roboflow project version ID
    version: int = 1
    # Model architecture type for Roboflow deploy (e.g., 'yolov8')
    model_type: str = "yolov8"
    # Path to model file (.pt) or training directory
    model_path: str = ""
    # Optional model weights filename (default is 'weights/best.pt')
    weights_filename: str | None = None


def upload_model_to_roboflow(cfg: UploadModelConfig) -> None:
    """Upload trained model weights to Roboflow version."""
    api_key = cfg.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("API key is required. Specify api_key in config or set ROBOFLOW_API_KEY env var.")

    if not cfg.project:
        raise ValueError("Roboflow 'project' ID must be specified.")

    path_obj = Path(cfg.model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Specified model path does not exist: {cfg.model_path}")

    # Determine model directory and weights filename
    if path_obj.is_file():
        model_dir = str(path_obj.parent)
        weights_file = cfg.weights_filename or path_obj.name
    else:
        model_dir = str(path_obj)
        weights_file = cfg.weights_filename

    print(f"\n--- Uploading Model Weights to Roboflow ---")
    print(f"Workspace:     {cfg.workspace or 'Default'}")
    print(f"Project ID:    {cfg.project}")
    print(f"Version ID:    {cfg.version}")
    print(f"Model Type:    {cfg.model_type}")
    print(f"Model Dir:     {model_dir}")
    if weights_file:
        print(f"Weights File:  {weights_file}")

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(cfg.workspace) if cfg.workspace else rf.workspace()
    proj = ws.project(cfg.project)
    proj_version = proj.version(cfg.version)

    if weights_file:
        proj_version.deploy(cfg.model_type, model_dir, weights_file)
    else:
        proj_version.deploy(cfg.model_type, model_dir)

    print("\nModel deploy to Roboflow completed successfully!")


@draccus.wrap()
def main(cfg: UploadModelConfig) -> None:
    """Main entrypoint for uploading model weights to Roboflow."""
    upload_model_to_roboflow(cfg)


if __name__ == "__main__":
    main()
