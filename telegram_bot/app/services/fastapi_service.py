import logging
import httpx
from config import FASTAPI_BACKEND_URL

logger = logging.getLogger(__name__)

class FastApiService:
    async def send_video_for_processing(
        self, 
        video_bytes: bytes, 
        original_filename: str,
        chat_id: int, 
        message_id: int
    ) -> tuple[bool, str | None]:
            
        files = {'video_file': (original_filename, video_bytes, 'video/mp4')}
        data = {'chat_id': str(chat_id), 'message_id': str(message_id)} 

        url = f"{FASTAPI_BACKEND_URL}/video/process"
        processing_message = None
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(url, data=data, files=files) 
            
            if response.status_code == 200 or response.status_code == 202:
                response_data = response.json()
                processing_message = response_data.get("message")
                logger.info(f"Запрос на обработку видео отправлен успешно. Ответ FastAPI: {processing_message}")
                return True, processing_message
            else:
                error_detail = response.text
                try:
                    error_detail = response.json().get("detail", response.text)
                except:
                    pass
                logger.error(f"Ошибка при отправке задачи на FastAPI: {response.status_code} - {error_detail}")
                return False, f"Ошибка сервера обработки ({response.status_code}): {error_detail}"
        except httpx.TimeoutException:
            logger.error(f"Таймаут при запросе к FastAPI: {url}")
            return False, "Сервер обработки не ответил вовремя. Попробуйте позже."
        except httpx.RequestError as e:
            logger.error(f"Ошибка HTTP запроса к FastAPI: {e}")
            return False, f"Ошибка соединения с сервером обработки: {e}"
        except Exception as e:
            logger.error(f"Неожиданная ошибка при запросе к FastAPI: {e}", exc_info=True)
            return False, f"Неожиданная ошибка связи: {str(e)}"

fastapi_service = FastApiService()