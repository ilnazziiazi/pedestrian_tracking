import os
import torch

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
YOLO_MODEL_FILENAME = os.getenv("YOLO_MODEL_FILENAME")
YOLO_MODEL_PATH = os.path.join("models/", YOLO_MODEL_FILENAME)

# Конфигурации для Ultralytics/YOLO
PEDESTRIAN_CLASSES = [4]
YOLO_CONFIDENCE_THRESHOLD = 0.3

TEMP_UPLOAD_DIR = "/app/temp_uploads"
ULTRALYTICS_OUTPUT_DIR = "/app/runs/track" 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(ULTRALYTICS_OUTPUT_DIR, exist_ok=True)