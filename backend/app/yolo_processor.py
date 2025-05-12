import logging
import os
import shutil
import uuid
import subprocess
from ultralytics import YOLO
from config import (
    YOLO_MODEL_PATH, DEVICE, PEDESTRIAN_CLASSES, 
    YOLO_CONFIDENCE_THRESHOLD, ULTRALYTICS_OUTPUT_DIR
)

logger = logging.getLogger(__name__)


def convert_avi_to_mp4_ffmpeg(avi_path: str) -> str:
    mp4_path = avi_path.replace(".avi", ".mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", avi_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            mp4_path
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    logger.info(f"Конвертация AVI → MP4 завершена: {mp4_path}")
    return mp4_path
        
        
class YoloVideoProcessor:
    def __init__(self):
        self.model = None
        self.device = DEVICE
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.model.to(self.device)
            logger.info(f"Модель YOLO ({YOLO_MODEL_PATH}) успешно загружена на {self.device}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели YOLO ({YOLO_MODEL_PATH}): {e}", exc_info=True)
            raise

    def process_and_track_video(self, input_video_path: str) -> tuple[str | None, int]:

        experiment_name = f"run_{uuid.uuid4().hex[:12]}"
        output_run_dir = os.path.join(ULTRALYTICS_OUTPUT_DIR, experiment_name) 
        
        processed_video_path = None
        unique_track_ids = set()

        try:
            logger.info(f"Начало трекинга видео: {input_video_path}")
            logger.info(f"Результаты будут сохранены в директорию внутри: {ULTRALYTICS_OUTPUT_DIR}/{experiment_name}")

            results_generator = self.model.track(
                source=input_video_path,
                tracker='bytetrack.yaml',
                save=True,
                save_txt=False,
                save_conf=False,
                classes=PEDESTRIAN_CLASSES,
                conf=YOLO_CONFIDENCE_THRESHOLD,
                project=ULTRALYTICS_OUTPUT_DIR,
                name=experiment_name,
                exist_ok=True,
                verbose=False,
                device=self.device,
                half=True, #FP16 для ускорения
                stream=True
            )

            for result in results_generator:
                if result.boxes is not None and result.boxes.id is not None:
                    current_ids = result.boxes.id.int().cpu().tolist()
                    unique_track_ids.update(current_ids)
                
            for f_name in os.listdir(output_run_dir):
                if f_name.lower().endswith(".avi"):
                    avi_path = os.path.join(output_run_dir, f_name)
                    processed_video_path = convert_avi_to_mp4_ffmpeg(avi_path)
                    break

            pedestrian_count = len(unique_track_ids)
            logger.info(f"Трекинг для '{input_video_path}' завершен. Уникальных пешеходов: {pedestrian_count}. "
                        f"Сохраненное видео: {processed_video_path}")
            
            return processed_video_path, pedestrian_count

        except Exception as e:
            logger.error(f"Ошибка во время трекинга YOLO для видео {input_video_path}: {e}", exc_info=True)
            if os.path.exists(output_run_dir):
                try:
                    shutil.rmtree(output_run_dir)
                    logger.info(f"Очищена папка эксперимента {output_run_dir} из-за ошибки.")
                except Exception as e_clean:
                    logger.error(f"Ошибка при очистке папки эксперимента {output_run_dir}: {e_clean}")
            return None, 0

# инициализация, чтобы модель загрузилась один раз при старте FastAPI приложения
try:
    yolo_video_processor = YoloVideoProcessor()
except Exception as e:
    logger.critical(f"Не удалось инициализировать YoloVideoProcessor: {e}", exc_info=True)