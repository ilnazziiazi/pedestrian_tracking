import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime 

LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 
    "logs"
    )

def get_logger(name):
    logger = logging.getLogger(name)
    
    # Проверяем, есть ли уже хендлеры у логгера
    if logger.handlers:
        return logger  # Возвращаем существующий логгер
        
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LOG_FORMAT)
    logger.addHandler(console_handler) 
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_file = os.path.join(
        LOG_DIR, 
        f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
    
    # Механизм ротации файла лога
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(LOG_FORMAT)
    logger.addHandler(file_handler)
    
    # Отключаем передачу логов родительским логгерам
    logger.propagate = False
    
    return logger