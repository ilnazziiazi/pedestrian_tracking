import asyncio
import logging
from aiogram import Bot, Dispatcher
from config import TELEGRAM_BOT_TOKEN
from handlers import user_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Запуск Telegram бота...")
    
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    dp.include_router(user_handlers.router)
    await bot.delete_webhook(drop_pending_updates=True)

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        logger.info("Telegram бот остановлен.")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Выключение бота.")