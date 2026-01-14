# Technical Documentation

## Project Structure

The project consists of three main components:

1. **Frontend (Streamlit)**
   - Located in `/frontend`
   - Main file: `client.py`
   - Provides an interactive interface for:
     - Data loading and EDA
     - Model training
     - Viewing model information
     - Obtaining predictions

2. **Backend service (FastAPI)**
   - Located in `/service`
   - Main file: `app.py`
   - Provides REST API endpoints for:
     - Data management
     - Model training
     - Model evaluation
     - Obtaining predictions

3. **ML pipeline**
   - Located in `/service/ml_pipeline.py`
   - Responsible for:
     - Image preprocessing
     - Feature extraction using ResNet50
     - SVM training
     - Model evaluation


## API Documentation

### Data Management
- `POST /upload_data`: Upload and unpack a dataset
- `GET /eda`: Get exploratory analysis results

### Model Management
- `POST /fit`: Train a new SVM model
- `GET /list_models`: List available models
- `POST /set_model`: Activate model for inference
- `POST /models_info`: Get detailed model metrics
- `DELETE /remove_all`: Delete all non-default models

### Inference
- `POST /predict`: Get predictions for uploaded image

## Streamlit Interface

The interface consists of 4 main sections:

1. **Data loading and EDA**
   - Loading a ZIP archive with a dataset
   - Viewing dataset statistics
   - Visualizing distributions

2. **Model training**
   - Configuring SVM hyperparameters
   - Training new models
   - Tracking the training process

3. **Model information**
   - Comparing models
   - Viewing metrics (Accuracy, Precision, Recall, F1)
   - Comparison of ROC and PR curves

4. **Model inference**
   - Selecting an active model
   - Loading images
   - Viewing predictions

## Quick Start

### Test Data
To test the functionality of the application, you can use data from the `/test_data` folder, which contains:
- Sample image for inference
- Test dataset in a ZIP archive for model training

### Start with Docker

1. Build images:
```bash
docker compose build
```
2. Start service:
```bash
docker compose up -d
```
### Start without Docker

1. Start backend service:
```bash
cd service
pip install -r requirements.txt
uvicorn app:app --reload
```
2. Start Streamlit frontend:
```bash
cd frontend
pip install -r requirements.txt
streamlit run client.py
```