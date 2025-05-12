from pathlib import Path
from uuid import uuid4
import pickle
import json
import cv2
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import asyncio
from fastapi import HTTPException
from data_loader import load_bounding_boxes, load_classes, get_files
import pandas as pd

# Предобученная Resnet для векторизации
resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Параметры ресайза
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#############################################
# Предобрабработка изображений для обучения #
#############################################
# Средний размер бибокса
async def get_average_bbox_size(images, labels_dir, class_names, class_name='person'):
    total_width, total_height, count = 0, 0, 0
    person_class_id = class_names.index(class_name)

    for image_name in images:
        bboxes = await load_bounding_boxes(image_name, labels_dir)
        bboxes = np.array(bboxes)
        if len(bboxes) == 0:
            continue

        mask = bboxes[:, 0] == person_class_id
        selected_bboxes = bboxes[mask]

        total_width += selected_bboxes[:, 3].sum()
        total_height += selected_bboxes[:, 4].sum()
        count += len(selected_bboxes)

    if count == 0:
        raise ValueError(f'Bboxes отсутствуют для класса {class_name}')

    avg_width = total_width / count
    avg_height = total_height / count
    return avg_width, avg_height


# Извлечение бибоксов
async def process_image(image_name, images_dir, labels_dir, class_names, avg_width,
                        avg_height, class_name='person', is_background=False, max_patches_per_image=3):
    person_class_id = class_names.index(class_name)
    image_path = images_dir / f"{image_name}.jpg"
    image = await asyncio.get_event_loop().run_in_executor(None, cv2.imread, str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    bboxes = await load_bounding_boxes(image_name, labels_dir)
    bboxes = np.array(bboxes)
    if len(bboxes) == 0:
        return []

    mask = bboxes[:, 0] == person_class_id
    selected_bboxes = bboxes[mask]

    if not is_background:
        return extract_person_patches(selected_bboxes, image, avg_width, avg_height, w, h)
    else:
        return extract_background_patches(selected_bboxes, image, avg_width, avg_height, w, h, max_patches_per_image)


# Извлекаем патчи с целевым классом
def extract_person_patches(selected_bboxes, image, avg_width, avg_height, w, h):
    x_min = ((selected_bboxes[:, 1] - avg_width / 2) * w).astype(int)
    y_min = ((selected_bboxes[:, 2] - avg_height / 2) * h).astype(int)
    x_max = ((selected_bboxes[:, 1] + avg_width / 2) * w).astype(int)
    y_max = ((selected_bboxes[:, 2] + avg_height / 2) * h).astype(int)

    x_min = np.clip(x_min, 0, w)
    y_min = np.clip(y_min, 0, h)
    x_max = np.clip(x_max, 0, w)
    y_max = np.clip(y_max, 0, h)

    patches = [image[y_min[i]:y_max[i], x_min[i]:x_max[i]] for i in range(len(selected_bboxes))]
    return patches


# Извлекаем патчи с фоном
def extract_background_patches(selected_bboxes, image, avg_width, avg_height, w, h, max_patches_per_image=3):
    person_boxes = np.array([
        (
            int((bbox[1] - bbox[3] / 2) * w),
            int((bbox[2] - bbox[4] / 2) * h),
            int((bbox[1] + bbox[3] / 2) * w),
            int((bbox[2] + bbox[4] / 2) * h),
        )
        for bbox in selected_bboxes
    ])

    patches = []
    attempts = 0

    while len(patches) < max_patches_per_image and attempts < 3:
        x_min = random.randint(0, w - int(avg_width * w))
        y_min = random.randint(0, h - int(avg_height * h))
        x_max = x_min + int(avg_width * w)
        y_max = y_min + int(avg_height * h)

        no_overlap = all(
            x_max <= px_min or x_min >= px_max or
            y_max <= py_min or y_min >= py_max
            for px_min, py_min, px_max, py_max in person_boxes
        )

        if no_overlap:
            patch = image[y_min:y_max, x_min:x_max]
            patches.append(patch)

        attempts += 1

    return patches


# Генератор загрузки
def create_dataloader(patches, label=None, indices=None, batch_size=512):
    transformed_patches = np.array([transform(patch).numpy() for patch in patches])
    tensor_X = torch.tensor(transformed_patches).float()

    if label is not None:
        labels = np.full(len(patches), label)
        tensor_y = torch.tensor(labels).long()
        dataset = TensorDataset(tensor_X, tensor_y)

    if indices is not None:
        tensor_indices = torch.tensor(indices).long()
        dataset = TensorDataset(tensor_X, tensor_indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


# Извлечение признаков из Resnet
def extract_resnet_features(dataloaders, resnet):
    resnet.eval()
    extracted_features = []
    extracted_labels_or_indices = []

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch_data in dataloader:
                images = batch_data[0]
                additional_data = batch_data[1]
                features = resnet(images).cpu().numpy()
                extracted_features.append(features)
                extracted_labels_or_indices.append(additional_data.cpu().numpy())

    return np.vstack(extracted_features), np.hstack(extracted_labels_or_indices)


# Главная функция с пайплайном процессинга
async def process_data(data_yaml_path, images_path, labels_path, image_files,
                       label_files, class_name='person', resnet=resnet_model):
    # Список классов
    class_names = await load_classes(data_yaml_path)
    # Средние размеры целевого бибокса
    avg_width, avg_height = await get_average_bbox_size(image_files, labels_path, class_names, class_name)
    # Извлечение патчей
    persons_tasks = [
        process_image(image_name, images_path, labels_path, class_names, avg_width, avg_height, class_name,
                      is_background=False)
        for image_name in image_files
    ]
    background_tasks = [
        process_image(image_name, images_path, labels_path, class_names, avg_width, avg_height, class_name,
                      is_background=True)
        for image_name in image_files
    ]
    persons_results = await asyncio.gather(*persons_tasks)
    backgrounds_results = await asyncio.gather(*background_tasks)
    persons_patches = [patch for result in persons_results for patch in result]
    background_patches = [patch for result in backgrounds_results for patch in result]
    # Создание даталоадеров
    persons_dataloader = create_dataloader(persons_patches, label=1)
    background_dataloader = create_dataloader(background_patches, label=0)
    # Выгрузка векторов
    X_person_features, y_person_labels = extract_resnet_features([persons_dataloader], resnet=resnet)
    X_background_features, y_background_labels = extract_resnet_features([background_dataloader], resnet=resnet)
    # Объединение данных
    X = np.vstack((X_person_features, X_background_features))
    y = np.hstack((y_person_labels, y_background_labels))
    X, y = shuffle(X, y, random_state=42)

    return X, y


#####################
# Обучение и оценка #
#####################
# Инициализация модели
def init_svm_model(hyperparams):
    return SVC(C=hyperparams.C, kernel=hyperparams.kernel, max_iter=hyperparams.max_iter, probability=True)


# Сохранение модели на диск
def save_model(model: SVC):
    model_id = str(uuid4())
    model_path = Path(f"./models/{model_id}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    return model_id, str(model_path)


# Оценка модели
def evaluate_model(model_id, model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        y_prob = model.decision_function(X_test)

        metrics = {
            "model_id": model_id,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}

        file_path = Path(f"./models_info/{model_id}.json")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with file_path.open("w") as file:
                json.dump(metrics, file)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка сохранения метрик качества модели {model_id}: {str(e)}"
            )

        if not file_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Файл {file_path} не существует после сохранения."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Произошла ошибка: {str(e)}."
        )


# Вызов процессинга, обучения и оценки
def processing_and_train(data_paths, hyperparams, queue):
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

        if len(X) < 50:
            valid_size = 0.5
        else:
            valid_size = 0.2

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=valid_size, random_state=42)

        try:
            svm = init_svm_model(hyperparams)
            svm.fit(X_train, y_train)

            model_id, model_path = save_model(svm)
            evaluate_model(model_id, svm, X_val, y_val)

            queue.put({
                "status": "success",
                "message": "Модель обучилась",
                "model_id": model_id,
                "model_path": model_path
            })

        except Exception as e:
            (
                queue.put({
                    "status": "error",
                    "message": f"Ошибка при передаче в очередь: {str(e)}"
                })
            )

    # Отлавливаем HTTPException, переданные вызываемыми внутри функциями
    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Произошла ошибка: {str(e)}."
        )


###################################################
# Обработка входного изображения для предсказания #
###################################################
# Кластеризация
def cluster_image(image, n_clusters=5):
    h, w, c = image.shape
    pixels = image.reshape((-1, c))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(pixels)

    clustered_image = kmeans.labels_.reshape(h, w)
    return clustered_image, kmeans


# Блюр изображений
def gaussian_blur(image, kernel_size=(15, 15), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)


# Выделение регионов интереса
def get_kmeans_regions(image, clustered_image, min_size=380, max_size=3200,
                       min_aspect_ratio=0.2, max_aspect_ratio=1, crop_ratio=0.1):
    regions = []
    img_h, img_w = image.shape[:2]

    center_x_min = int(img_w * crop_ratio)
    center_y_min = int(img_h * crop_ratio)
    center_x_max = int(img_w * (1 - crop_ratio))
    center_y_max = int(img_h * (1 - crop_ratio))

    for label in np.unique(clustered_image):
        mask = (clustered_image == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if not (center_x_min <= x <= center_x_max and
                    center_y_min <= y <= center_y_max and
                    center_x_min <= x + w <= center_x_max and
                    center_y_min <= y + h <= center_y_max):
                continue

            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            if (min_size <= area <= max_size and
                    min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                region = image[y:y + h, x:x + w]
                regions.append((region, (x, y, w, h)))

    return regions


# Процессинг изображения для предсказания
def process_inference_image(image, n_clusters=5, blur_kernel=(15, 15), blur_sigma=0,
                            min_size=380, max_size=3200, min_aspect_ratio=0.2, max_aspect_ratio=1,
                            crop_ratio=0.1, resnet=resnet_model):
    # Конвертация в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Размытие
    blurred = gaussian_blur(image, kernel_size=blur_kernel, sigma=blur_sigma)
    # Кластеризация
    clustered_image, _ = cluster_image(blurred, n_clusters=n_clusters)
    # Получение регионов
    regions = get_kmeans_regions(
        blurred, clustered_image,
        min_size=min_size, max_size=max_size,
        min_aspect_ratio=min_aspect_ratio, max_aspect_ratio=max_aspect_ratio,
        crop_ratio=crop_ratio
    )

    if not regions:
        raise ValueError("Не найдено ни одного региона.")

    # Присваиваем регионам индексы
    patches, indices, bboxes = zip(*[
        (region, idx, coords) for idx, (region, coords) in enumerate(regions)
    ])
    # Создаем DataLoader
    dataloader = create_dataloader(patches, indices)
    # Извлекаем признаки
    features, extracted_indices = extract_resnet_features([dataloader], resnet=resnet)

    df = pd.DataFrame({
        "index": extracted_indices,
        "patch": list(patches),
        "bbox": list(bboxes),
        "features": list(features)
    })

    return df
