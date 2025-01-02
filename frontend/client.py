import streamlit as st
import asyncio
import httpx
import matplotlib.pyplot as plt

async def eda(file):
    """Получение EDA."""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/eda", files={"file": (file.name, file.getvalue(), file.type)})
        return response.json()

BASE_URL = "http://127.0.0.1:8000"

# Заголовок
st.title("Проект по трекингу пешеходов")

st.sidebar.title("Станицы")
page = st.sidebar.radio(" ", ["Демонстрация EDA", "Предсказание"])

if page == "Демонстрация EDA":
    st.header("Разведочный анализ данных")
    # Загрузка файла
    file = st.file_uploader("Выберите файл", type=["zip"])
    container = st.empty()
    if file is not None:
        try:
            st.title("Демонстрация EDA")
            results = asyncio.run(eda(file)) 
            
            classes = list(results['image_count'].keys())
            counts = list(results['image_count'].values())

            fig, ax = plt.subplots()
            fig.set_figheight(7)
            fig.set_figwidth(7)
            plt.bar(classes, counts, color='blue', edgecolor='white')
            plt.xlabel('Class')
            plt.ylabel(f'Image Count')
            plt.title(f'Distribution of Image Count per Class')
            plt.xticks(rotation=45, ha='right')
    
            plt.tight_layout()
            plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            st.pyplot(fig)
            #st.write('По графику видно. Что ...')

        except Exception as e:
            container.error(f"Ошибка: {str(e)}")


elif page == "Предсказание":
    st.header("Предсказание")
    st.write("Загрузка файла")

    # Загрузка файла
    file = st.file_uploader("Выбери файл", type=["jpg", "png", "jpeg"])
    if file is not None:
        st.image(file, caption="Загруженное изображение", use_container_width=True)
