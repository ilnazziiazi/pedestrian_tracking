import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from utils.utils import get_logger


logger = get_logger("streamlit_client")

logger.info("Запуск приложения")

BASE_URL = "http://127.0.0.1:8000"
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
DEFAULT_TIMEOUT = 10
DEFAULT_MAX_ITER = 1000


def setup_layout_template() -> dict:
    """Create common layout template for plots."""
    return {
        'height': 500,
        'margin': dict(t=100),
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': dict(size=12),
        'title': {
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16)
        }
    }


def create_histogram(data: list, title: str, x_label: str, layout_template: dict) -> go.Figure:
    layout = layout_template.copy()
    layout['title']['text'] = title

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data[0],
        name='С пешеходами',
        opacity=0.75,
        nbinsx=50,
        marker_color='#1f77b4'
    ))
    fig.add_trace(go.Histogram(
        x=data[1],
        name='Без пешеходов',
        opacity=0.75,
        nbinsx=50,
        marker_color='#2ca02c'
    ))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title='Количество',
        barmode='overlay',
        **layout,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    return fig


# Сайдбар для выбора действия
st.sidebar.title("Проект по детекции и трекингу пешеходов")
action = st.sidebar.selectbox(
    "Выберите действие",
    ["Загрузка данных и EDA", "Обучение модели", "Информация о модели", "Инференс модели"]
)
logger.info(f"Выбрано действие: {action}")

# Загрузка данных
if action == "Загрузка данных и EDA":
    logger.info("Открыт раздел 'Загрузка данных и EDA'")
    st.header("Загрузка данных и EDA")
    uploaded_file = st.file_uploader("Выберите ZIP архив с данными", type="zip")

    if uploaded_file is not None:
        logger.info(f"Загружен файл: {uploaded_file.name}, размер: {uploaded_file.size} байт")

        if st.button("Загрузить данные"):
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
            with st.spinner("Анализ данных..."):
                try:
                    response = requests.get(f"{BASE_URL}/eda")
                    if response.status_code == 200:
                        result = response.json()
                        logger.info("Получены результаты EDA")

                        # Процент изображений с людьми
                        st.subheader("Статистика датасета")
                        people_presence = result['people_presence']
                        st.metric(
                            "Процент изображений с пешеходами",
                            f"{people_presence:.1%}"
                        )

                        # Распределение размеров изображений
                        st.subheader("Распределение размеров изображений")

                        # Ширина
                        fig_width = create_histogram(
                            result['dist_img_width_groups'],
                            'Распределение ширины изображений',
                            'Ширина изображения',
                            setup_layout_template()
                        )
                        st.plotly_chart(fig_width, use_container_width=True)

                        # Высота
                        fig_height = create_histogram(
                            result['dist_img_heights_groups'],
                            'Распределение высоты изображений',
                            'Высота изображения',
                            setup_layout_template()
                        )
                        st.plotly_chart(fig_height, use_container_width=True)

                        # Соотношение сторон
                        fig_ratio = create_histogram(
                            result['dist_img_ratios_groups'],
                            'Распределение соотношения сторон',
                            'Соотношение сторон',
                            setup_layout_template()
                        )
                        st.plotly_chart(fig_ratio, use_container_width=True)

                        # Тепловые карты
                        fig_heat = go.Figure()

                        # Все изображения
                        fig_heat.add_trace(
                            go.Heatmap(
                                z=result['hitmap_all'],
                                colorscale='Hot',
                                name='Все изображения',
                                showscale=True
                            )
                        )

                        # Только изображения с пешеходами
                        fig_heat.add_trace(
                            go.Heatmap(
                                z=result['hitmap_person'],
                                colorscale='Hot',
                                name='Только изображения с пешеходами',
                                visible=False,
                                showscale=True
                            )
                        )

                        # Добавляем горизонтальный переключатель с увеличенным отступом от заголовка
                        fig_heat.update_layout(
                            title={
                                'text': 'Тепловые карты расположения пешеходов',
                                'y': 0.95,  # Поднимаем заголовок выше
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            height=600,
                            updatemenus=[{
                                'type': 'buttons',
                                'direction': 'right',
                                'showactive': True,
                                'buttons': [
                                    {'label': 'Все изображения',
                                     'method': 'update',
                                     'args': [{'visible': [True, False]}]},
                                    {'label': 'Только с пешеходами',
                                     'method': 'update',
                                     'args': [{'visible': [False, True]}]}
                                ],
                                'x': 0.1,
                                'y': 1.08,  # Опускаем переключатели ниже
                                'xanchor': 'left',
                                'yanchor': 'top'
                            }]
                        )

                        st.plotly_chart(fig_heat, use_container_width=True)

                    else:
                        st.error(f"Ошибка при получении EDA: {response.text}")

                except Exception as e:
                    logger.error(f"Ошибка при получении EDA: {str(e)}")
                    st.error(f"Ошибка при получении EDA: {str(e)}")


# Обучение модели
elif action == "Обучение модели":
    logger.info("Открыт раздел 'Обучение модели'")
    st.header("Обучение модели")

    # Параметры SVM
    st.subheader("Параметры модели")
    C = st.slider("C", 0.1, 10.0, 1.0, 0.1)
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    max_iter = st.number_input("Max iterations", 100, 10000, 1000, 100)
    timeout = st.number_input("Timeout (seconds)", 5, 300, 10, 5)

    if st.button("Обучить модель"):
        logger.info(f"Запущено обучение модели с параметрами: C={C}, kernel={kernel}, max_iter={max_iter}, timeout={timeout}")
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


# Информация о модели
elif action == "Информация о модели":
    logger.info("Открыт раздел 'Информация о модели'")
    st.header("Информация о моделях")

    response = requests.get(f"{BASE_URL}/list_models")
    models = response.json()
    logger.info(f"Получен список моделей: {len(models)} моделей")

    if models:
        model_options = [model['id'] for model in models]

        selected_models = st.multiselect(
            "Выберите модели для сравнения",
            options=model_options,
            default=model_options[0] if model_options else None
        )

        if selected_models:
            logger.info(f"Запрошена информация о моделях: {selected_models}")
            response = requests.post(
                f"{BASE_URL}/models_info",
                json={"ids": selected_models}
            )

            if response.status_code == 200:
                models_info = response.json()

                metrics_data = []
                for model_info in models_info:
                    metrics = model_info['metrics']
                    metrics_row = {
                        "ID модели": model_info['id'],
                        "Accuracy": f"{metrics['accuracy']:.3f}",
                        "Precision": f"{metrics['precision']:.3f}",
                        "Recall": f"{metrics['recall']:.3f}",
                        "F1-score": f"{metrics['f1_score']:.3f}",
                        "ROC AUC": f"{metrics['roc_auc']:.3f}",
                        "PR AUC": f"{metrics['pr_auc']:.3f}"
                    }
                    metrics_data.append(metrics_row)

                st.subheader("Метрики качества")
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True)

                # ROC кривые
                st.subheader("ROC кривые")
                fig_roc = go.Figure()
                for model_info in models_info:
                    metrics = model_info['metrics']
                    fig_roc.add_trace(go.Scatter(
                        x=metrics['roc_curve']['y'],
                        y=metrics['roc_curve']['x'],
                        name=f"{model_info['id']} (AUC={metrics['roc_auc']:.3f})"
                    ))

                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    line=dict(dash='dash', color='gray'),
                    showlegend=False
                ))

                fig_roc.update_layout(
                    title='ROC Curves',
                    width=800,
                    height=600,
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    xaxis=dict(constrain='domain'),
                    hovermode='closest',
                    legend=dict(
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        orientation="h"
                    )
                )
                st.plotly_chart(fig_roc, use_container_width=True)

                # PR кривые
                st.subheader("PR кривые")
                fig_pr = go.Figure()
                for model_info in models_info:
                    metrics = model_info['metrics']
                    fig_pr.add_trace(go.Scatter(
                        x=metrics['pr_curve']['y'],
                        y=metrics['pr_curve']['x'],
                        name=f"{model_info['id']} (AUC={metrics['pr_auc']:.3f})"
                    ))

                fig_pr.update_layout(
                    title='Precision-Recall Curves',
                    width=800,
                    height=600,
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    xaxis=dict(constrain='domain'),
                    hovermode='closest',
                    legend=dict(
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        orientation="h"
                    )
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.error(f"Ошибка при получении информации о моделях: {response.text}")
    else:
        st.info("Нет доступных моделей. Сначала обучите модель.")


# Инференс модели
elif action == "Инференс модели":
    logger.info("Открыт раздел 'Инференс модели'")
    st.header("Инференс модели")

    response = requests.get(f"{BASE_URL}/list_models")
    models = response.json()
    logger.info(f"Получен список моделей для инференса: {len(models)} моделей")

    if models:
        table_data = []
        model_options = []  # Список для selectbox

        for model in models:
            model_row = {
                "ID модели": str(model['id']),
                "C параметр": str(model['data']['hyperparams'].get('C', '')),
                "Ядро": str(model['data']['hyperparams'].get('kernel', '')),
                "Макс. итераций": str(model['data']['hyperparams'].get('max_iter', ''))
            }
            table_data.append(model_row)
            model_options.append(model['id'])

        df = pd.DataFrame(table_data)
        st.subheader("Список доступных моделей")
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            selected_model = st.selectbox(
                "Выберите модель для активации",
                options=model_options
            )

            if st.button("Активировать выбранную модель"):
                logger.info(f"Попытка активации модели {selected_model}")
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

        st.subheader("Предсказание")

        if not st.session_state['model_activated']:
            st.warning("Для получения предсказаний необходимо активировать модель")
        else:
            uploaded_image = st.file_uploader(
                "Загрузите изображение для предсказания", 
                type=ALLOWED_IMAGE_TYPES
            )

            if uploaded_image is not None:
                logger.info(f"Загружено изображение для предсказания: {uploaded_image.name}, размер: {uploaded_image.size} байт")
                image = Image.open(uploaded_image)
                orig_width, orig_height = image.size
                logger.info(f"Размеры изображения: {orig_width}x{orig_height}")

                if st.button("Получить предсказание"):
                    uploaded_image.seek(0)

                    try:
                        with st.spinner("Получение предсказания..."):
                            files = {"image_file": uploaded_image}
                            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=5)
                            response.raise_for_status()

                            if response.status_code == 200:
                                result = response.json()

                                if result['data']:
                                    uploaded_image.seek(0)
                                    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                                    image_cv = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                                    img_height, img_width = image_cv.shape[:2]

                                    for pred in result['data']:
                                        bbox = pred['bbox']
                                        prob = pred['probability']

                                        x, y, w, h = map(int, bbox)

                                        x1 = int((x / img_width) * orig_width)
                                        y1 = int((y / img_height) * orig_height)
                                        x2 = int(((x + w) / img_width) * orig_width)
                                        y2 = int(((y + h) / img_height) * orig_height)

                                        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                        text = f"{prob:.2%}"
                                        cv2.putText(image_cv, text, (x1, y1-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                                    st.image(image_rgb, caption="Результат детекции", use_container_width=True)
                                    st.success(f"Обнаружено пешеходов: {len(result['data'])}")
                                else:
                                    st.info("Пешеходы на изображении не обнаружены")
                            else:
                                st.error(f"Ошибка при получении предсказания: {response.text}")

                    except requests.Timeout:
                        logger.error("Timeout error during prediction request")
                        st.error("Превышено время ожидания ответа от сервера")
                    except requests.RequestException as e:
                        logger.error(f"Request error during prediction: {str(e)}")
                        st.error(f"Ошибка при отправке запроса: {str(e)}")
    else:
        st.info("Нет доступных моделей. Сначала обучите модель.")