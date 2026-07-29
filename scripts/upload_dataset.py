import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
import draccus
from roboflow import Roboflow
from tqdm import tqdm


@dataclass
class RoboflowConfig:
    """Dataclass configuration for Roboflow upload."""

    # Roboflow private API key
    api_key: str | None = None
    # Roboflow workspace ID
    workspace: str = ""
    # Roboflow project ID
    project: str = ""
    # Upload batch name
    batch_name: str = ""
    # Number of retry attempts for failed uploads
    num_retries: int = 3


@dataclass
class GDriveConfig:
    """Dataclass configuration for Google Drive upload."""

    # Google Drive folder ID
    folder_id: str = ""
    # Path to credentials JSON file
    credentials_file: str = "credentials.json"
    # Zip dataset directory before uploading
    zip_before_upload: bool = True


@dataclass
class UploadConfig:
    """Dataclass configuration for multi-target dataset upload."""

    # Target platform: 'roboflow', 'gdrive', or 'all'
    target: str = ""
    # Path to YOLO dataset directory
    dataset_dir: str = ""
    # Roboflow settings
    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    # Google Drive settings
    gdrive: GDriveConfig = field(default_factory=GDriveConfig)


def upload_to_roboflow(cfg: RoboflowConfig, dataset_path: Path) -> None:
    """Upload dataset images and labels to Roboflow."""
    api_key = cfg.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("API key is required. Specify api_key in config or set ROBOFLOW_API_KEY env var.")

    if not cfg.workspace or not cfg.project:
        raise ValueError("Both 'workspace' and 'project' must be specified in config.")

    print(f"\n--- Uploading to Roboflow (Workspace: {cfg.workspace}, Project: {cfg.project}) ---")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(cfg.workspace).project(cfg.project)

    splits = ["train", "valid", "test"]
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total_uploaded = 0

    for split in splits:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            print(f"Directory missing for split '{split}': {img_dir}, skipping.")
            continue

        image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in supported_exts]
        print(f"Uploading split '{split}': {len(image_files)} images found.")

        for img_path in tqdm(image_files, desc=f"Roboflow Upload ({split})"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            ann_path_str = str(lbl_path) if lbl_path.exists() else None

            try:
                project.upload(
                    image_path=str(img_path),
                    annotation_path=ann_path_str,
                    split=split,
                    batch_name=cfg.batch_name,
                    num_retry_uploads=cfg.num_retries,
                )
                total_uploaded += 1
            except Exception as e:
                print(f"Failed to upload {img_path.name}: {e}")

    print(f"Roboflow upload complete! Total images uploaded: {total_uploaded}")


def upload_to_gdrive(cfg: GDriveConfig, dataset_path: Path) -> None:
    """Upload dataset archive or files to Google Drive."""
    print(f"\n--- Uploading to Google Drive (Folder ID: {cfg.folder_id or 'Root'}) ---")
    
    upload_file_path = dataset_path

    # Compress dataset directory if requested
    if cfg.zip_before_upload:
        zip_path = dataset_path.parent / f"{dataset_path.name}"
        expected_zip = Path(f"{zip_path}.zip")

        if expected_zip.exists() and expected_zip.stat().st_size > 0:
            print(f"Zip archive already exists at: {expected_zip}. Skipping compression.")
            upload_file_path = expected_zip
        else:
            print(f"Compressing dataset to: {expected_zip} ...")
            archive_file = shutil.make_archive(str(zip_path), "zip", dataset_path)
            upload_file_path = Path(archive_file)
            print(f"Dataset compressed successfully ({upload_file_path.stat().st_size / (1024*1024):.2f} MB)")

    # Attempt upload using google-api-python-client
    try:
        import json
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2 import service_account

        cred_path = Path(cfg.credentials_file)
        if not cred_path.exists():
            print(f"Warning: Credentials file not found at '{cred_path}'. Skipping API authentication.")
            print(f"File ready for manual upload at: {upload_file_path}")
            return

        with open(cred_path, "r") as f:
            cred_data = json.load(f)

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        creds = None

        if "installed" in cred_data or "web" in cred_data:
            try:
                from google_auth_oauthlib.flow import InstalledAppFlow
                from google.auth.transport.requests import Request
                from google.oauth2.credentials import Credentials

                token_path = cred_path.parent / "token.json"
                if token_path.exists():
                    creds = Credentials.from_authorized_user_file(str(token_path), scopes)

                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), scopes)
                        creds = flow.run_local_server(port=0)

                    with open(token_path, "w") as token_file:
                        token_file.write(creds.to_json())
            except ImportError:
                print("Missing 'google-auth-oauthlib' package for OAuth login. Run: pip install google-auth-oauthlib")
                return
        elif cred_data.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_file(str(cred_path), scopes=scopes)
        else:
            raise ValueError(f"Unrecognized credentials format in '{cred_path}'.")

        service = build("drive", "v3", credentials=creds)

        file_metadata = {"name": upload_file_path.name}
        if cfg.folder_id:
            file_metadata["parents"] = [cfg.folder_id]

        media = MediaFileUpload(str(upload_file_path), resumable=True)
        file_obj = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        print(f"Successfully uploaded to Google Drive! File ID: {file_obj.get('id')}")

    except ImportError:
        print(f"Notice: Google API client library ('google-api-python-client') not installed.")
        print(f"Dataset is ready for upload at: {upload_file_path}")
    except Exception as e:
        print(f"Error during Google Drive upload: {e}")


@draccus.wrap()
def main(cfg: UploadConfig) -> None:
    """Main execution entrypoint for multi-target dataset upload."""
    dataset_path = Path(cfg.dataset_dir).resolve()
    target = cfg.target.lower()

    if target in ("roboflow", "all"):
        upload_to_roboflow(cfg.roboflow, dataset_path)

    if target in ("gdrive", "googledrive", "all"):
        upload_to_gdrive(cfg.gdrive, dataset_path)

    if target not in ("roboflow", "gdrive", "googledrive", "all"):
        raise ValueError(f"Invalid target '{cfg.target}'. Must be 'roboflow', 'gdrive', or 'all'.")


if __name__ == "__main__":
    main()
