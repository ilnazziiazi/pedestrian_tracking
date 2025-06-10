import logging
import os
import uuid
import shutil
import time
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from contextlib import asynccontextmanager

from schemas import ProcessInitiatedResponse
from config import TEMP_UPLOAD_DIR, ULTRALYTICS_OUTPUT_DIR, DEVICE
from yolo_processor import yolo_video_processor
from telegram_sender import send_message, send_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


async def background_video_processing_task(
    temp_input_video_path: str, 
    chat_id: int, 
    original_status_message_id: int
):
    task_start_time = time.time()
    logger.info(f"Начало обработки видео {temp_input_video_path} для chat_id {chat_id}")
    

    processed_video_path: str | None = None
    experiment_output_dir: str | None = None

    try:
        processed_video_path, pedestrian_count, trajectories, boxes_per_frame = (yolo_video_processor.process_and_track_video(temp_input_video_path))
        final_video_path = processed_video_path

        caption = (
            f"✅ Видео обработано!\n"
            f"🚶‍♂️ Уникальных пешеходов: {pedestrian_count}\n"
        )

        if trajectories and boxes_per_frame:
            try:
                clustered_video_path, cluster_counts = yolo_video_processor.clusters_visualize(
                    input_video_path=temp_input_video_path,
                    trajectories=trajectories,
                    boxes_per_frame=boxes_per_frame,
                    output_path=os.path.dirname(processed_video_path)
                )
                if clustered_video_path:
                    final_video_path = clustered_video_path

                noise_tracks = cluster_counts.get(-1, 0)
                total_tracks = sum(cluster_counts.values())
                if total_tracks > 0:
                    noise_ratio = (noise_tracks / total_tracks) * 100
                else:
                    noise_ratio = 0.0

                clusters_info = "\n".join(
                    f"Кластер {cid+1}: число пешеходов {count}"
                    for cid, count in sorted(cluster_counts.items())
                    if cid != -1
                )

                if clusters_info:
                    caption += f"{clusters_info}\n"
                caption += f"Пешеходов, не попавших в кластеры: {noise_tracks} ({noise_ratio:.1f}% от общего числа)\n"

            except ValueError as ve:
                logger.warning(f"Кластеризация не выполнена: {ve}")
                caption += "⚠️ Не удалось выделить кластеры движения.\n"
                final_video_path = processed_video_path

        task_duration = time.time() - task_start_time
        logger.info(f"Видео {temp_input_video_path} обработано за {task_duration:.2f} сек.")
        caption += f"⏱ Время обработки видео: {task_duration:.1f} сек."

        send_success = await send_video(chat_id, final_video_path, caption)
        if not send_success:
            await send_message(chat_id, "Не удалось отправить обработанное видео. Пожалуйста, попробуйте позже или проверьте настройки бота.")

    except Exception as e:
        task_duration = time.time() - task_start_time
        logger.error(f"Ошибка при обработке видео {temp_input_video_path} за {task_duration:.2f} сек: {e}", exc_info=True)
        await send_message(chat_id, f"⚠️ Произошла ошибка сервера во время обработки видео. Попробуйте позже.", message_id_to_edit=original_status_message_id)
    
    finally:
        if os.path.exists(temp_input_video_path):
            try:
                os.remove(temp_input_video_path)
                logger.info(f"Временный загруженный файл {temp_input_video_path} удален.")
            except Exception as e_clean:
                logger.error(f"Ошибка при удалении временного загруженного файла {temp_input_video_path}: {e_clean}")
        
        if processed_video_path:
            experiment_output_dir = os.path.dirname(processed_video_path)
            if experiment_output_dir and os.path.exists(experiment_output_dir) and ULTRALYTICS_OUTPUT_DIR in experiment_output_dir: # Двойная проверка
                try:
                    shutil.rmtree(experiment_output_dir)
                    logger.info(f"Папка результатов Ultralytics {experiment_output_dir} удалена.")
                except Exception as e_clean_exp:
                    logger.error(f"Ошибка при удалении папки результатов Ultralytics {experiment_output_dir}: {e_clean_exp}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Backend запускается...")
    logger.info(f"Используемое устройство для PyTorch: {DEVICE}")

    # Проверка инициализации YoloVideoProcessor
    if yolo_video_processor is not None:
        logger.info("YoloVideoProcessor успешно инициализирован.")
    else:
        raise RuntimeError("YoloVideoProcessor failed to initialize. Check model path and CUDA setup.")
    yield


app = FastAPI(
    title="Backend API",
    lifespan=lifespan
)

@app.post("/api/v1/video/process", response_model=ProcessInitiatedResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    chat_id: int = Form(...),
    message_id: int = Form(...)
):
    if not video_file.content_type == "video/mp4":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Неверный тип файла. Требуется MP4.")

    # Сохранение загруженного файла во временную папку
    file_extension = ".mp4" # Принудительно, т.к. мы проверяем content_type
    temp_input_filename = f"{uuid.uuid4()}{file_extension}"
    temp_input_video_path = os.path.join(TEMP_UPLOAD_DIR, temp_input_filename)

    try:
        with open(temp_input_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        logger.info(f"Видеофайл '{video_file.filename}' сохранен временно как '{temp_input_video_path}' для обработки.")
    except Exception as e:
        logger.error(f"Не удалось сохранить загруженный видеофайл '{video_file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка сохранения видеофайла на сервере.")
    finally:
        await video_file.close()

    background_tasks.add_task(
        background_video_processing_task, 
        temp_input_video_path, 
        chat_id, 
        message_id
    )
    
    logger.info(f"Фоновая задача для обработки видео '{temp_input_video_path}' (chat_id: {chat_id}) добавлена.")
    
    return ProcessInitiatedResponse(
        message=f"Видео '{video_file.filename or 'файл'}' принято и поставлено в очередь на обработку. Вы получите уведомление о завершении."
    )

@app.get("/health")
async def health_check():
    yolo_status = "инициализирован" if yolo_video_processor and yolo_video_processor.model else "НЕ инициализирован"
    return {
        "status": "healthy", 
        "pytorch_device": str(DEVICE),
        "yolo_model_status": yolo_status
        }