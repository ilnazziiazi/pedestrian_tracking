import multiprocessing
import asyncio
import uvicorn
import shutil
import pickle
import yaml
import time
import os
import zipfile
from multiprocessing import Process, Queue
from pydantic import BaseModel, RootModel
from typing import Literal, Dict, Any, List
from fastapi import UploadFile, FastAPI, HTTPException, File
from data_loader import save_and_unpack
from preprocessing import process_data
from sklearn.svm import SVC
from http import HTTPStatus
from pathlib import Path
from uuid import uuid4
from collections import defaultdict


app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json"
)


models: Dict[str, Any] = {}
loaded_models: Dict[str, bool] = {}
uploaded_data_paths: Dict[str, Path] = {}


# Базовая директория
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


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



# Функции
def load_classes(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def find_jpg_files_without_extension(folder_path):
    # Проверяем, существует ли путь
    if not os.path.exists(folder_path):
        print(f"Путь {folder_path} не существует.")
        return []
    
    # Получаем список всех файлов и папок в указанной директории
    all_items = os.listdir(folder_path)
    
    # Фильтруем только файлы с расширением .jpg и удаляем расширение
    jpg_files = [
        os.path.splitext(item)[0] for item in all_items
        if os.path.isfile(os.path.join(folder_path, item)) and item.lower().endswith('.jpg')
    ]
    
    return jpg_files


# Функции
def init_svm_model(hyperparams: SVMHyper) -> SVC:
    return SVC(C=hyperparams.C, kernel=hyperparams.kernel, max_iter=hyperparams.max_iter)

def get_files(path, extension):
    return [file.stem for file in path.glob(f"*.{extension}")]

def save_model(model: SVC):
    model_id = str(uuid4())
    model_path = Path(f"./models/{model_id}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    return model_id, str(model_path)

def processing_and_train(data_paths: Dict[str, Path], hyperparams: SVMHyper, queue: Queue):
    try:
        data_yaml_path = data_paths["data_yaml_path"]
        images_path = data_paths["images_path"]
        labels_path = data_paths["labels_path"]

        image_files = get_files(images_path, "jpg")
        label_files = get_files(labels_path, "txt")

        X, y = asyncio.run(process_data(
            data_yaml_path,
            images_path,
            labels_path,
            image_files,
            label_files
        ))

        svm = init_svm_model(hyperparams)
        svm.fit(X, y)

        model_id, model_path = save_model(svm)

        queue.put({
            "status": "success",
            "message": "Модель обучилась",
            "model_id": model_id,
            "model_path": model_path
        })

    except Exception as e:
        queue.put({
            "status": "error",
            "message": f"Ошибка во время обработки и обучения: {str(e)}"
        })

# Эндпоинты
@app.post("/upload", response_model=UploadResponse)
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

        num_images = len(list(images_path.glob("*.jpg")))

        return UploadResponse(
            status="success",
            message="Данные загружены",
            data={"num_images": num_images}
        )

    except HTTPException as e:
        return UploadResponse(
            status="error",
            message=e.detail,
            data={}
        )

    except Exception as e:
        return UploadResponse(
            status="error",
            message=f"Ошибка: {str(e)}",
            data={}
        )

@app.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest) -> FitResponse:
    try:
        data_dir = Path("./uploaded_data")
        if not data_dir.exists() or not any(data_dir.iterdir()):
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Сначала загрузите данные")

        start = time.time()

        queue = Queue()
        process = Process(target=processing_and_train, args=(uploaded_data_paths, request.hyperparams, queue))
        process.start()

        process.join(timeout=request.timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return FitResponse(
                status="error",
                message=f"Превышен лимит {request.timeout} секунд — процесс прерван."
            )

        if not queue.empty():
            result = queue.get()
            if result["status"] == "success":
                models[result["model_id"]] = {
                    "path": result["model_path"],
                    "hyperparams": request.hyperparams.dict()
                }
        else:
            result = {
                "status": "error",
                "message": "Процесс завершился без результата."
            }

        end = time.time()
        total_duration = end - start

        return FitResponse(
            **result,
            duration=total_duration
        )

    except HTTPException as e:
        return FitResponse(
            status="error",
            message=e.detail
        )

    except Exception as e:
        return FitResponse(
            status="error",
            message=f"Ошибка: {str(e)}"
        )

@app.get("/list_models", response_model=ModelListResponse)
async def list_models()-> ModelListResponse:
    model_list = [ModelListItem(id=model_id,
                                data=ModelData(path=str(models[model_id]["path"]),
                                               hyperparams=models[model_id]["hyperparams"])
                                )  for model_id in models.keys()
                  ]
    return ModelListResponse(root=model_list)


@app.post("/set_models", response_model=SetResponse)
async def set_models(request: SetRequest) -> SetResponse:
    model_id = request.id
    if model_id not in models.keys():
        raise HTTPException(
            status_code=422,
            detail=[{"loc": ["body", "id"], "msg": f"Модель '{model_id}' не существует", "type": "value_error"}]
        )

    # Деактивируем другие модели, если они были активны
    for loaded_model in loaded_models.keys():
        loaded_models[loaded_model] = False

    loaded_models[model_id] = True
    return SetResponse(message=f"Модель '{model_id}' загружена")



@app.post("/eda")
async def eda(file: UploadFile = File(...)) -> Dict:
    """
    Exploratory Data Analysis (EDA) на основе загруженного архива изображений.
    """
    try:
        # 1. Сохраняем zip-файл и распаковываем
        zip_path = os.path.join(CURRENT_DIR, file.filename)
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(CURRENT_DIR)

        destination_dir = CURRENT_DIR + '/data'
        #destination_dir.mkdir(exist_ok=True)

        data_yaml_path = destination_dir + '/data.yaml'
        class_names = load_classes(data_yaml_path)

        train_images_dir = destination_dir + '/images'
        train_labels_dir = destination_dir + '/labels'

        train_images = find_jpg_files_without_extension(train_images_dir)
        class_image_count = defaultdict(int)
        class_bbox_count = defaultdict(int)

        for image_name in train_images:
            label_file = train_labels_dir + f'/{image_name}.txt'
            
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
    
        return {'filename': CURRENT_DIR, 
                'image_count': dict(class_image_count), 
                'bbox_count': dict(class_bbox_count)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
