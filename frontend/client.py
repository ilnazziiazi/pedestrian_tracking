import sys
import streamlit as st
import matplotlib.pyplot as plt
import requests
import pandas as pd

sys.path.append('../')
from utils.utils import get_logger
logger = get_logger("streamlit_client")

logger.info("Запуск приложения")

BASE_URL = "http://127.0.0.1:8000"


# Сайдбар для выбора действия
st.sidebar.title("Проект по детекции и трекингу пешеходов")
action = st.sidebar.selectbox(
    "Выберите действие",
    ["Загрузка данных и EDA", "Обучение модели", "Выбор модели и предсказание"]
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


# Выбор модели и предсказание
elif action == "Выбор модели и предсказание":
    st.header("Выбор активной модели")
    
    if st.button("Получить список моделей"):
        response = requests.get(f"{BASE_URL}/list_models")
        models = response.json()
        
        if models:
            table_data = []
            for model in models:
                model_row = {
                    "model_id": model['id'],
                    "c_parameter": model['data']['hyperparams'].get('C', ''),
                    "kernel": model['data']['hyperparams'].get('kernel', ''),
                    "max_iter": model['data']['hyperparams'].get('max_iter', '')
                }
                table_data.append(model_row)
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Нет доступных моделей")
    
    model_id = st.text_input("Введите ID модели")
    
    if st.button("Активировать модель"):
        if model_id:
            response = requests.post(
                f"{BASE_URL}/set_model",
                json={"id": model_id}
            )
            result = response.json()
            st.success(result["message"])
        else:
            st.warning("Введите ID модели")
