FROM python:3.10

COPY requirements.txt app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "client.py"]