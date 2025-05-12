from pydantic import BaseModel, Field

class ProcessInitiatedResponse(BaseModel):
    message: str = Field("Задача на обработку видео принята и запущена в фоновом режиме.",
                         example="Видео принято, начинаем обработку...")