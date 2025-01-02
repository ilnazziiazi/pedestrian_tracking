import multiprocessing
import asyncio
import uvicorn
import shutil
import pickle
import time
from multiprocessing import Process, Queue
from pydantic import BaseModel, RootModel, Field
from typing import Literal, Dict, Any, List
from fastapi import UploadFile, FastAPI, HTTPException, File
from data_loader import save_and_unpack, load_classes
from ml_pipeline import get_files, processing_and_train, process_inference_image
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from src_eda import *
import json
import joblib
import pandas as pd

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json"
)

models: Dict[str, Any] = {}
loaded_models: Dict[str, bool] = {}
uploaded_data_paths: Dict[str, Path] = {}
all_images: List = []
images_with_person: List = []
eda_data: Dict[str, Any] = {}
class_image_count = defaultdict(int)
class_bbox_count = defaultdict(int)

# Классы Request-Response
# Гиперпараметры модели обучения
class SVMHyper(BaseModel):
    C: float = 1.0
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "linear"
    max_iter: int = 1000

# Upload архива
class UploadResponse(BaseModel):
    status: str
    message: str
    data: dict

# Fit
class FitRequest(BaseModel):
    hyperparams: SVMHyper
    timeout: int = 10

class FitResponse(BaseModel):
    status: str
    message: str
    duration: float = None
    model_id: str = None

# Set
class SetRequest(BaseModel):
    id: str

class SetResponse(BaseModel):
    message: str

# Model List
class ModelData(BaseModel):
    path: str
    hyperparams: dict

class ModelListItem(BaseModel):
    id: str
    data: ModelData

class ModelListResponse(RootModel[List[ModelListItem]]):
    pass

# Model Info
class ModelsInfoRequest(BaseModel):
    ids: List[str] = Field(default=list)

class Curves(BaseModel):
    x: List[float] = Field(default=list)   # Ось X
    y: List[float] = Field(default=list)    # Ось Y

class TrainEvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    roc_curve: Curves
    pr_curve: Curves

class ModelsInfoResponseItem(BaseModel):
    id: str
    metrics: TrainEvaluationMetrics

class ModelsInfoResponse(RootModel[List[ModelsInfoResponseItem]]):
    pass

# Predict
class PredictionItem(BaseModel):
    index: int
    bbox: Any
    probability: Any

class PredictResponse(BaseModel):
    model_id: str
    data: List[PredictionItem]

# EDA
class EDARequest(BaseModel):
    pass

class EDAResponse(BaseModel):
    status: str
    message: str
    data_to_plot: dict


# Эндпоинты
@app.post("/upload_data", response_model=UploadResponse)
async def upload_data(archive: UploadFile) -> UploadResponse:
    try:
        output_dir = Path("./uploaded_data")
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        data_yaml_path, images_path, labels_path = await save_and_unpack(archive, "./uploaded_data")

        uploaded_data_paths.update({
            "data_yaml_path": data_yaml_path,
            "images_path": images_path,
            "labels_path": labels_path
        })

        all_images.append(get_image_names(images_path))
        class_names = load_classes(data_yaml_path)
        person_class_id = class_names.index('person')

        for image_name in all_images:
            label_file = (labels_path if image_name in all_images else None) / f'{image_name}.txt'

            if label_file.exists():
                with open(label_file, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        if int(parts[0]) == person_class_id:
                            images_with_person.append(image_name)
                            break

            classes_in_image = set()
            try:
                with open(label_file, 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        class_name = class_names[class_id]

                        class_bbox_count[class_name] += 1

                        if class_name not in classes_in_image:
                            class_image_count[class_name] += 1
                            classes_in_image.add(class_name)

            except Exception as e:
                print(f'Ошибка при обработке файла {image_name}: {e}')
                continue

        num_images = len(list(images_path.glob("*.jpg")))

        eda_data["class_image_count"] = class_image_count
        eda_data["class_bbox_count"] = class_bbox_count

        return UploadResponse(
            status="success",
            message="Данные загружены",
            data={"num_images": num_images}
        )

    # Отлавливаем HTTPException, переданные вызываемыми внутри функциями
    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Произошла ошибка: {str(e)}."
        )

    # Отлавливаем HTTPException, переданные вызываемыми внутри функциями
    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Произошла ошибка: {str(e)}."
        )

@app.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest) -> FitResponse:
    try:
        data_dir = Path("./uploaded_data")
        if not data_dir.exists() or not any(data_dir.iterdir()):
            raise HTTPException(
                status_code=400,
                detail="Сначала загрузите данные."
            )

        start = time.time()

        queue = Queue()
        process = Process(target=processing_and_train, args=(uploaded_data_paths, request.hyperparams, queue))
        process.start()

        process.join(timeout=request.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            raise HTTPException(
                status_code=408,
                detail=f"Превышен лимит обучения {request.timeout} секунд — процесс прерван. Попробуйте увеличить лимит."
            )

        if not queue.empty():
            result = queue.get()
            if result["status"] == "success":
                if "models" not in globals():
                    raise HTTPException(
                        status_code=500,
                        detail="Глобальный cловарь models недоступен."
                    )
                models[result["model_id"]] = {
                    "path": result["model_path"],
                    "hyperparams": request.hyperparams.dict()
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result["message"]
                )
        else:
            raise HTTPException(
                status_code=500,
                detail="Процесс завершился без результата."
            )

        end = time.time()
        total_duration = end - start

        return FitResponse(
            **result,
            duration=total_duration
        )

    # Отлавливаем HTTPException, переданные вызываемыми внутри функциями
    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Произошла ошибка: {str(e)}."
        )

@app.get("/list_models", response_model=ModelListResponse)
async def list_models()-> ModelListResponse:
    model_list = [ModelListItem(id=model_id,
                                data=ModelData(path=str(models[model_id]["path"]),
                                               hyperparams=models[model_id]["hyperparams"])
                                )  for model_id in models.keys()
                  ]
    return ModelListResponse(root=model_list)

@app.post("/set_model", response_model=SetResponse)
async def set_model(request: SetRequest) -> SetResponse:
    model_id = request.id
    if model_id not in models.keys():
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{model_id}' не существует."
        )

    # Деактивируем другие модели, если они были активны
    for loaded_model in loaded_models.keys():
        loaded_models[loaded_model] = False

    loaded_models[model_id] = True
    return SetResponse(message=f"Модель '{model_id}' загружена.")

@app.post("/predict", response_model=PredictResponse)
async def predict(image_file: UploadFile) -> PredictResponse:
    image_bytes = np.asarray(bytearray(image_file.file.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Загруженный файл не является изображением."
        )

    df_processed = process_inference_image(image)
    loaded_model = next((key for key, value in loaded_models.items() if value), None)

    if loaded_model is None:
        raise HTTPException(
            status_code=400,
            detail="Активных моделей не найдено. Используйте метод set_model, чтобы активировать модель."
        )

    model_path = Path(f"./models/{loaded_model}.pkl")
    svm = joblib.load(model_path)
    probas = svm.predict_proba(np.vstack(df_processed["features"]))[:, 1]
    df_processed["probability"] = probas

    df_person = df_processed[df_processed["probability"] > 0.85].reset_index(drop=True)

    df_person["patch"] = df_person["patch"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    response_json = df_person[["index", "bbox", "probability"]].to_dict(orient="records")

    return PredictResponse(
        model_id=loaded_model,
        data=response_json
    )

@app.post("/models_info", response_model=ModelsInfoResponse)
async def models_info(request: ModelsInfoRequest) -> ModelsInfoResponse:
    models_info_path = Path("models_info")

    if not models_info_path.exists() or not models_info_path.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Директория {models_info_path} не существует. Сначала обучите модель."
        )

    # Если приходит пустой список, возвращаем инфо для всех моделей
    # Такая реализация нужна для сравнения нескольких моделей на клиенте
    if not request.ids:
        model_ids = [file.stem for file in models_info_path.glob("*.json")]
    else:
        model_ids = request.ids

    models_info_json = []
    for model_id in model_ids:
        file_path = models_info_path / f"{model_id}.json"
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Файл метрик для модели {model_id} не найден. Сначала обучите модель."
            )

        with file_path.open("r") as file:
            data = json.load(file)

            metrics = TrainEvaluationMetrics(
                accuracy=data["accuracy"],
                precision=data["precision"],
                recall=data["recall"],
                f1_score=data["f1_score"],
                roc_auc=data["roc_auc"],
                pr_auc=data["pr_auc"],
                roc_curve=Curves(x=data["roc_curve"]["tpr"], y=data["roc_curve"]["fpr"]),
                pr_curve=Curves(x=data["pr_curve"]["recall"], y=data["pr_curve"]["precision"])
            )

            models_info_json.append(ModelsInfoResponseItem(id=data["model_id"], metrics=metrics))

    return ModelsInfoResponse(models_info_json)

@app.post("/eda", response_model=EDAResponse)
async def eda(request: EDARequest) -> EDAResponse:
    try:
        data_yaml_path = uploaded_data_paths["data_yaml_path"]
        images_path = uploaded_data_paths["images_path"]
        labels_path = uploaded_data_paths["labels_path"]

        image_files = get_files(images_path, "jpg")
        label_files = get_files(labels_path, "txt")

        class_names = load_classes(data_yaml_path)

        all_img_size_stats, all_width_list, all_height_list, all_proportions_list = get_image_size(all_images)
        person_img_size_stats, person_width_list, person_height_list, person_proportions_list = get_image_size(all_images, images_with_person)
        all_img_count = all_img_size_stats.get('image_count')
        person_img_count = person_img_size_stats.get('image_count')

        # 1
        eda_data["get_people_presence"] = f"Люди присутствуют на {round(person_img_count/all_img_count,2)*100}% изображений"

        # 2
        images_wo_person = [img for img in all_images if img not in (images_with_person or [])]
        no_person_img_size_stats, no_person_width_list, no_person_height_list, no_person_proportions_list \
            = get_image_size(all_images, images_wo_person)
        eda_data["distrib_img_width_groups"] = (person_width_list, no_person_width_list)

        # 3
        eda_data["distrib_img_heights_groups"] = (person_height_list, no_person_height_list)

        # 4
        eda_data["distrib_img_ratios_groups"] = (person_proportions_list, no_person_proportions_list)

        # 5
        # Check upload data

        # 6
        eda["hitmap_all"] = get_bboxes_heatmap(all_images)
        eda["hitmap_person"] = get_bboxes_heatmap(images_with_person)

    except Exception as e:
        return EDAResponse(
            status="Error",
            message=f"Ошибка: {str(e)}"
        )


    if loaded_model is None:
        raise HTTPException(
            status_code=400,
            detail="Активных моделей не найдено. Используйте метод set_model, чтобы активировать модель."
        )

    model_path = Path(f"./models/{loaded_model}.pkl")
    svm = joblib.load(model_path)
    probas = svm.predict_proba(np.vstack(df_processed["features"]))[:, 1]
    df_processed["probability"] = probas

    df_person = df_processed[df_processed["probability"] > 0.85].reset_index(drop=True)

    df_person["patch"] = df_person["patch"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    response_json = df_person[["index", "bbox", "probability"]].to_dict(orient="records")

    return PredictResponse(
        model_id=loaded_model,
        data=response_json
    )