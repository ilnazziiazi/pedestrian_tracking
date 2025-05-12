# Проект по трекингу пешеходов
## Описание проекта
В рамках проекта планируется создать сервис, который будет определять пешеходов на видео, рассчитывать пешеходный трафик, его среднюю скорость и плотность распределения в течение дня, а также людей пешеходов по полу, возрасту и благосостоянию. Такая аналитика может быть полезной для понимания целевой аудитории компаниям, размещающих рекламу на улице.

## Документация
- [Техническая документация](./report.md)
- [Описание датасета](./dataset.md)
- [Результаты EDA](./EDA.md)
- [Чекпоинты разработки](./checkpoints.md)

## Для запуска необходимо:
1. Добавить в корень проекта файл .env с `TELEGRAM_BOT_TOKEN=...`
2. Установить [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) для работы с GPU.
3. Добавить [веса](https://drive.google.com/file/d/1L8H0u8CHvF3oKpP4jApN2guUblrAq0sW/view?usp=share_link) модели в папку `models/yolov11.pt`
4. Выполнить `docker compose up --build`

## Демонстрация сервиса (telegram bot)
Есть вероятность, что хост с ботом в данный момент онлайн. 
Можно попробовать написать [этому боту](https://t.me/pedestrian_tracking_bot)
> Ограничение на вес видеофайла - 20 Мб.

## Демонстрация сервиса для работы с моделями и их обучения (old_service)

[![Демонстрация проекта](https://img.youtube.com/vi/38uSmBjsTpY/0.jpg)](https://www.youtube.com/watch?v=38uSmBjsTpY)

## Состав команды
**Куратор команды** — Соборнов Тимофей @saintedts
|Имя|Telegram|Github|
|---|---|---|
|Брежнева Ангелина|@eslaxg|brezhnevaan|
|Зиязиев Ильназ|@ilnazzia|ilnazzia|
|Петров Андрей|@andreyabox|andreyabox|
|Янышев Дмитрий|@yanyshev_dima|yanyshev|

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
