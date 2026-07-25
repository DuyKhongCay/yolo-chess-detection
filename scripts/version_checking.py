from ultralytics import YOLO

# Load the model
model = YOLO("/home/duykhongcay/lerobot_ws/AI_Chess/runs/detect/train/weights/best.pt")

# Access the raw checkpoint dictionary
ckpt = model.ckpt

# 1. Get the ultralytics library version used for training
# This indicates the YOLO suite version (e.g., '8.2.50' is YOLOv8, '8.3.0' is YOLO11)
ultralytics_version = ckpt.get("version")
print(f"Ultralytics version: {ultralytics_version}")

# 2. Get the base model configuration / weights file used initially
train_args = ckpt.get("train_args", {})
base_model = train_args.get("model")
print(f"Base model: {base_model}")

# 3. Check the task type (e.g., 'detect', 'segment', 'classify')
print(f"Task: {model.task}")
