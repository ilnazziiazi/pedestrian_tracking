import logging
import httpx
import os
from config import TELEGRAM_BOT_TOKEN

logger = logging.getLogger(__name__)

API_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

async def send_message(chat_id: int, text: str, message_id_to_edit: int = None) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен. Сообщение не может быть отправлено.")
        return False
    
    url = f"{API_BASE_URL}/editMessageText" if message_id_to_edit else f"{API_BASE_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if message_id_to_edit:
        payload["message_id"] = message_id_to_edit

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=15.0)
            response.raise_for_status()
            logger.info(f"Telegram: Сообщение отправлено/изменено для chat_id {chat_id}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Telegram API ошибка (статус {e.response.status_code}): {e.response.text} для chat_id {chat_id}")
    except httpx.RequestError as e:
        logger.error(f"Telegram API ошибка запроса: {e} для chat_id {chat_id}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при отправке сообщения в Telegram: {e}", exc_info=True)
    return False

async def send_video(chat_id: int, video_path: str, caption: str = "") -> bool:
    if not os.path.exists(video_path):
        logger.error(f"Видеофайл не найден для отправки: {video_path}")
        return False

    url = f"{API_BASE_URL}/sendVideo"
    try:
        with open(video_path, "rb") as video_file:
            files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
            data = {"chat_id": chat_id, "caption": caption}
            async with httpx.AsyncClient(timeout=300.0) as client: # 5 минут таймаут
                response = await client.post(url, data=data, files=files)
                response.raise_for_status()
            logger.info(f"Telegram: Видео успешно отправлено для chat_id {chat_id}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Telegram API ошибка при отправке видео (статус {e.response.status_code}): {e.response.text} для chat_id {chat_id}")
    except httpx.RequestError as e:
        logger.error(f"Telegram API ошибка запроса при отправке видео: {e} для chat_id {chat_id}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка при отправке видео в Telegram: {e}", exc_info=True)
    return False