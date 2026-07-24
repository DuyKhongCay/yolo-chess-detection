import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Google Drive URL for pre-processed ChessReD dataset images
IMAGES_URL = "https://drive.google.com/file/d/1jxmFxjOy0qefdCZ_x3DMNtsvAK4LojEw/view"

# 4TU.ResearchData URL for ChessReD annotations.json
ANNOTATIONS_URL = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/3cae6364-daca-4967-b426-1e4b68cdb64c"

# Target directory to save and extract dataset
TARGET_DIR = Path(__file__).resolve().parent / "datasets"


def ensure_gdown():
    """Ensure gdown package is installed to handle Google Drive downloads."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("Package 'gdown' not found. Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        return gdown


def download_annotations(url: str, output_dir: Path) -> None:
    """Check local existence and download annotations.json if needed.

    Args:
        url (str): Download URL for annotations.json.
        output_dir (Path): Directory where annotations.json should be saved.
    """
    annotations_path = output_dir / "annotations.json"
    if annotations_path.exists() and annotations_path.stat().st_size > 0:
        print(f"'annotations.json' already exists at: {annotations_path}. Skipping download.")
        return

    print(f"Downloading annotations.json from: {url}")
    output_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, annotations_path)
    print(f"'annotations.json' downloaded successfully to: {annotations_path}")


def download_and_extract_images(url: str, output_dir: Path) -> None:
    """Check local existence, download dataset zip file, and extract images if needed.

    Args:
        url (str): Google Drive file URL for images dataset.
        output_dir (Path): Output directory where dataset will be stored.
    """
    images_dir = output_dir / "images"

    # Check if images directory exists and contains data
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Dataset images already exist at: {images_dir}. Skipping image download and extraction.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_file_path = output_dir / "preprocessed_images.zip"

    # Download zip file if missing or empty
    if not zip_file_path.exists() or zip_file_path.stat().st_size == 0:
        gdown = ensure_gdown()
        print(f"Downloading images dataset from Google Drive: {url}")
        downloaded_path = gdown.download(url, str(zip_file_path))
        if not downloaded_path or not os.path.exists(downloaded_path):
            raise RuntimeError("Download failed. Please check the Google Drive link or network connection.")
        print(f"Dataset downloaded successfully to: {downloaded_path}")
    else:
        print(f"Zip file already exists at: {zip_file_path}. Skipping download.")

    print(f"Extracting dataset to: {output_dir}")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        try:
            from tqdm import tqdm
            for file_name in tqdm(file_list, desc="Extracting", unit="file"):
                zip_ref.extract(file_name, output_dir)
        except ImportError:
            zip_ref.extractall(output_dir)

    print(f"Extraction completed successfully! Images are ready at: {images_dir}")


def download_chessred(images_url: str, annotations_url: str, output_dir: Path) -> None:
    """Main workflow to check local dataset existence and download missing files.

    Args:
        images_url (str): URL for images zip file.
        annotations_url (str): URL for annotations.json file.
        output_dir (Path): Path to dataset directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download annotations.json if missing
    download_annotations(annotations_url, output_dir)

    # Step 2: Download and extract images if missing
    download_and_extract_images(images_url, output_dir)

    print(f"\nChessReD dataset is ready at: {output_dir}")


if __name__ == "__main__":
    download_chessred(IMAGES_URL, ANNOTATIONS_URL, TARGET_DIR)
