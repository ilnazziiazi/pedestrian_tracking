import sys
import streamlit as st
import matplotlib.pyplot as plt
import requests
import pandas as pd
from PIL import Image
import cv2
import numpy as np

sys.path.append('../')
from utils.utils import get_logger
logger = get_logger("streamlit_client")

logger.info("Запуск приложения")

BASE_URL = "http://127.0.0.1:8000"


# Сайдбар для выбора действия
st.sidebar.title("Проект по детекции и трекингу пешеходов")
action = st.sidebar.selectbox(
    "Выберите действие",
    ["Загрузка данных и EDA", "Обучение модели", "Информация о модели", "Инференс модели"]
)
logger.info(f"Выбрано действие: {action}")

# Загрузка данных
if action == "Загрузка данных и EDA":
    st.header("Загрузка данных и EDA")
    uploaded_file = st.file_uploader("Выберите ZIP архив с данными", type="zip")
    
    if uploaded_file is not None:
        logger.info(f"Загружен файл: {uploaded_file.name}")
        
        if st.button("Загрузить"):
            files = {"archive": uploaded_file}
            with st.spinner("Загрузка данных..."):
                try:
                    response = requests.post(f"{BASE_URL}/upload_data", files=files)
                    result = response.json()
                    
                    if result["status"] == "success":
                        logger.info("Данные успешно загружены")
                        st.success(f"Данные успешно загружены! {result['message']}")
                        st.write(f"Количество изображений: {result['data']['num_images']}")
                    else:
                        logger.error(f"Ошибка при загрузке: {result['message']}")
                        st.error(f"Ошибка при загрузке: {result['message']}")
                except Exception as e:
                    logger.error(f"Ошибка при отправке запроса: {str(e)}")
                    st.error(f"Ошибка при отправке запроса: {str(e)}")

        if st.button("Получить EDA"):
            files = {"file": uploaded_file}
            with st.spinner("Анализ данных..."):
                try:
                    response = requests.post(f"{BASE_URL}/eda", files=files)
                    result = response.json()
                    logger.info("Получены результаты EDA")
                    
                    st.subheader("Результаты EDA")
                    
                    # Визуализация количества изображений по классам
                    fig, ax = plt.subplots(figsize=(10, 6))
                    classes = list(result['image_count'].keys())
                    counts = list(result['image_count'].values())
                    ax.bar(classes, counts)
                    plt.xticks(rotation=45, ha='right')
                    plt.title("Количество изображений по классам")
                    st.pyplot(fig)
                    
                    # Визуализация количества bbox по классам
                    fig, ax = plt.subplots(figsize=(10, 6))
                    classes = list(result['bbox_count'].keys())
                    counts = list(result['bbox_count'].values())
                    ax.bar(classes, counts)
                    plt.xticks(rotation=45, ha='right')
                    plt.title("Количество bbox по классам")
                    st.pyplot(fig)
                except Exception as e:
                    logger.error(f"Ошибка при получении EDA: {str(e)}")
                    st.error(f"Ошибка при получении EDA: {str(e)}")


# Обучение модели
elif action == "Обучение модели":
    st.header("Обучение модели")
    
    # Параметры SVM
    st.subheader("Параметры модели")
    C = st.slider("C", 0.1, 10.0, 1.0, 0.1)
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    max_iter = st.number_input("Max iterations", 100, 10000, 1000, 100)
    timeout = st.number_input("Timeout (seconds)", 5, 300, 10, 5)
    
    if st.button("Обучить модель"):
        params = {
            "hyperparams": {
                "C": C,
                "kernel": kernel,
                "max_iter": max_iter
            },
            "timeout": timeout
        }
        
        with st.spinner("Обучение модели..."):
            response = requests.post(f"{BASE_URL}/fit", json=params)
            result = response.json()
            
            if result["status"] == "success":
                st.success(f"Модель успешно обучена! ID модели: {result['model_id']}")
                st.write(f"Время обучения: {result['duration']:.2f} секунд")
            else:
                st.error(f"Ошибка при обучении: {result['message']}")


# Инференс модели
elif action == "Инференс модели":
    st.header("Инференс модели")
    
    response = requests.get(f"{BASE_URL}/list_models")
    models = response.json()
    
    if models:
        table_data = []
        model_options = []  # Список для selectbox
        
        for model in models:
            model_row = {
                "ID модели": model['id'],
                "C параметр": model['data']['hyperparams'].get('C', ''),
                "Ядро": model['data']['hyperparams'].get('kernel', ''),
                "Макс. итераций": model['data']['hyperparams'].get('max_iter', '')
            }
            table_data.append(model_row)
            model_options.append(model['id'])
        
        # Показываем таблицу моделей
        df = pd.DataFrame(table_data)
        st.subheader("Список доступных моделей")
        st.dataframe(df, use_container_width=True)
        
        # Создаем колонки для разделения интерфейса
        col1, col2 = st.columns(2)
        
        with col1:
            # Выбор модели через selectbox
            selected_model = st.selectbox(
                "Выберите модель для активации",
                options=model_options
            )
            
            # Кнопка активации выбранной модели
            if st.button("Активировать выбранную модель"):
                response = requests.post(
                    f"{BASE_URL}/set_model",
                    json={"id": selected_model}
                )
                result = response.json()
                st.session_state['model_activated'] = True
                st.session_state['active_model'] = selected_model
        
        with col2:
            if 'model_activated' not in st.session_state:
                st.session_state['model_activated'] = False
            
            if st.session_state['model_activated']:
                st.success(f"Активная модель: {st.session_state['active_model']}")
            else:
                st.warning("Модель не активирована")
        
        # Добавляем секцию для загрузки изображения и получения предсказаний
        st.subheader("Предсказание")
        
        if not st.session_state['model_activated']:
            st.warning("Для получения предсказаний необходимо активировать модель")
        else:
            uploaded_image = st.file_uploader("Загрузите изображение для предсказания", type=["jpg", "jpeg", "png"])
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                
                if st.button("Получить предсказание"):
                    uploaded_image.seek(0)
                    
                    try:
                        files = {"image_file": uploaded_image}
                        response = requests.post(f"{BASE_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            if result['data']:
                                # Конвертируем PIL Image в массив numpy
                                uploaded_image.seek(0)
                                image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                                image_cv = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                                
                                # Рисуем bbox для каждого обнаруженного пешехода
                                for pred in result['data']:
                                    bbox = pred['bbox']
                                    prob = pred['probability']
                                    
                                    # Получаем координаты bbox
                                    x1, y1, x2, y2 = map(int, bbox)
                                    
                                    # Рисуем прямоугольник
                                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Добавляем текст с вероятностью
                                    text = f"{prob:.2%}"
                                    cv2.putText(image_cv, text, (x1, y1-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                # Конвертируем BGR в RGB для корректного отображения
                                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                                
                                # Показываем изображение с bbox
                                st.image(image_rgb, caption="Результат детекции", use_container_width=True)
                                st.success(f"Обнаружено пешеходов: {len(result['data'])}")
                            else:
                                st.info("Пешеходы на изображении не обнаружены")
                        else:
                            st.error(f"Ошибка при получении предсказания: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Ошибка при обработке запроса: {str(e)}")
    else:
        st.info("Нет доступных моделей. Сначала обучите модель.")
