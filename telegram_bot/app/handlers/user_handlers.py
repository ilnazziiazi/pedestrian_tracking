import logging
import uuid
import io
from aiogram import Router, F, Bot
from aiogram.filters import CommandStart
from aiogram.types import Message

from services.fastapi_service import fastapi_service
from config import MAX_DURATION_SECONDS, MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)
router = Router()

@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! 👋 Отправь мне видео (MP4), и я:\n"
        "🤖 Обнаружу пешеходов\n"
        "👣 Отслежу их пути\n"
        "📊 Сообщу их общее количество\n\n"
    )

@router.message(F.video)
async def handle_video(message: Message, bot: Bot):
    if message.video is None:
        await message.reply("Произошла ошибка с видеофайлом. Попробуйте еще раз.")
        return

    if message.video.mime_type != "video/mp4":
        await message.reply("Пожалуйста, отправьте видео в формате MP4.")
        return
    
    if message.video.duration > MAX_DURATION_SECONDS:
        await message.reply(f"Видео слишком длинное. Пожалуйста, отправьте видео длительностью до {MAX_DURATION_SECONDS // 60} минут.")
        return

    if message.video.file_size > MAX_FILE_SIZE_MB * 1024 * 1024 :
        await message.reply(f"Видеофайл слишком большой (макс. {MAX_FILE_SIZE_MB}MB). Пожалуйста, отправьте файл меньшего размера.")
        return

    status_message = await message.reply("Получил видео. Отправляю на сервер для обработки... ⏳")
    
    try:
        video_file_id = message.video.file_id
        file_info = await bot.get_file(video_file_id)
        
        if file_info.file_path is None:
            await status_message.edit_text("Не удалось получить путь к файлу для скачивания. Попробуйте еще раз.")
            return

        downloaded_file_stream: io.BytesIO = await bot.download_file(file_info.file_path)
        video_bytes_to_send = downloaded_file_stream.getvalue()
        downloaded_file_stream.close()
        
        original_filename_for_server = f"{uuid.uuid4()}.mp4" 
        
        success, server_response_message = await fastapi_service.send_video_for_processing(
            video_bytes=video_bytes_to_send,
            original_filename=original_filename_for_server,
            chat_id=message.chat.id, 
            message_id=status_message.message_id 
        )

        if success:
            await status_message.edit_text(
                f"{server_response_message or 'Видео принято в обработку.'}\n"
                "Я пришлю результат, как только он будет готов. Обычно это занимает некоторое время в зависимости от длины видео. 🕒"
            )
        else:
            await status_message.edit_text(
                f"Не удалось отправить видео на обработку: {server_response_message or 'Неизвестная ошибка сервера.'} 😥"
            )

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке видео от пользователя {message.chat.id}: {e}", exc_info=True)
        await status_message.edit_text("Произошла внутренняя ошибка при подготовке видео. Попробуйте позже. 😥")

@router.message(F.animation)
async def handle_animation_as_video(message: Message, bot: Bot):
    class MockVideo:
        def __init__(self, file_id, mime_type, duration, file_size):
            self.file_id = file_id
            self.mime_type = mime_type
            self.duration = duration
            self.file_size = file_size

    actual_mime_type = "video/mp4"

    if message.document and message.document.mime_type == "video/mp4":
         actual_mime_type = message.document.mime_type
         if message.document.file_size: actual_file_size = message.document.file_size
    elif message.animation.mime_type == "video/mp4":
         actual_mime_type = message.animation.mime_type
    
    if actual_mime_type != "video/mp4":
        await message.reply("Этот GIF не в формате MP4. Пожалуйста, отправьте видео в формате MP4.")
        return

    mock_video_message = message.model_copy()
    mock_video_message.video = MockVideo(
        message.animation.file_id, 
        actual_mime_type, 
        message.animation.duration,
        actual_file_size
    )
    await handle_video(mock_video_message, bot)


@router.message()
async def handle_other_messages(message: Message):
    await message.reply("Я умею обрабатывать только видеофайлы в формате MP4. Пожалуйста, отправьте видео.")