from pydantic import BaseModel, Field
from fastapi import UploadFile, FastAPI, HTTPException
from data_loader import save_and_unpack
from preprocessing import process_data
from sklearn.svm import SVC
import numpy as np
import time
from http import HTTPStatus
from uvicorn import run
app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json"
)

svm = SVC(kernel="linear", C=1.0, probability=True)

@app.post("/fit")
async def fit(archive: UploadFile):
    try:
        start = time.time()
        data_yaml_path, images_path, labels_path, image_files, label_files = save_and_unpack(archive, "./uploaded_data")
        X, y = process_data(data_yaml_path, images_path, labels_path, image_files, label_files)
        svm.fit(X, y)

        end = time.time()
        duration = end - start

        num_samples = int(len(X))
        num_features = int(len(X[0]))
        unique_classes = list(map(int, np.unique(y)))

        return {
            "status": "success",
            "message": "Модель успешно обучена",
            "data": {
                "num_samples": num_samples,
                "num_features": num_features,
                "classes": unique_classes
            },
            "train_duration": duration
        }
    except HTTPException as e:
        return {"status": "error", "message": e.detail}
    except Exception as e:
        return {"status": "error", "message": f"Ошибка: {str(e)}"}