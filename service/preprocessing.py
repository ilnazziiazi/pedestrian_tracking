from data_loader import load_bounding_boxes, load_classes
import cv2
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from sklearn.utils import shuffle
import asyncio

# Предобученная Resnet для векторизации
resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Параметры ресайза
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
async def process_image(image_name, images_dir, labels_dir, class_names, avg_width, avg_height, class_name='person', is_background=False, max_patches_per_image=3):
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
def create_dataloader(patches, label, batch_size=512):
    labels = np.full(len(patches), label)

    transformed_patches = np.array([transform(patch).numpy() for patch in patches])

    tensor_X = torch.tensor(transformed_patches).float()
    tensor_y = torch.tensor(labels).long()
    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

# Извлечение признаков из Resnet
def extract_resnet_features(dataloaders, resnet):
    resnet.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for dataloader in dataloaders:
            for images, labels in dataloader:
                features = resnet(images)

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    return np.vstack(all_features), np.hstack(all_labels)


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
