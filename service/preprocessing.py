import os
import zipfile
from pathlib import Path
from data_loader import load_bounding_boxes, load_classes
import cv2
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import numpy as np
from sklearn.utils import shuffle

# Предобученная resnet для векторизации
resnet_model = models.resnet50(pretrained=True)

# Параметры ресайза
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

# Средний размер bbox-а
def get_average_bbox_size(images, labels_dir, class_names, class_name='person'):
    total_width, total_height, count = 0, 0, 0
    person_class_id = class_names.index(class_name)

    for image_name in images:
        bboxes = load_bounding_boxes(image_name, labels_dir)
        for class_id, x_center, y_center, box_width, box_height in bboxes:
            if class_id == person_class_id:
                total_width += box_width
                total_height += box_height
                count += 1

    if count == 0:
        raise ValueError(f'Bboxes отсутствуют для класса {class_name}')

    avg_width = total_width / count
    avg_height = total_height / count
    return avg_width, avg_height

# Извлекаем патчи с целевым классом
def extract_persons(images_dir, images, labels_dir, avg_width, avg_height, class_names, class_name='person'):
    person_class_id = class_names.index(class_name)
    persons_patches = []

    for image_name in images:
        image_path = images_dir / f"{image_name}.jpg"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        bboxes = load_bounding_boxes(image_name, labels_dir)
        for class_id, x_center, y_center, box_width, box_height in bboxes:
            if class_id == person_class_id:
                x_min = int((x_center - avg_width / 2) * w)
                y_min = int((y_center - avg_height / 2) * h)
                x_max = int((x_center + avg_width / 2) * w)
                y_max = int((y_center + avg_height / 2) * h)

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                patch = image[y_min:y_max, x_min:x_max]
                persons_patches.append(patch)

    return persons_patches

# Извлекаем патчи фона
def extract_background(images_dir, images, labels_dir, avg_width, avg_height, class_names, class_name='person', max_patches_per_image=3):
    background_patches = []
    person_class_id = class_names.index(class_name)

    for image_name in images:
        image_path = images_dir / f"{image_name}.jpg"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Загружаем bboxes
        bboxes = load_bounding_boxes(image_name, labels_dir)
        person_boxes = [
            (int((x_center - box_width / 2) * w),
             int((y_center - box_height / 2) * h),
             int((x_center + box_width / 2) * w),
             int((y_center + box_height / 2) * h))
            for class_id, x_center, y_center, box_width, box_height in bboxes
            if class_id == person_class_id
        ]

        patches_found = 0
        attempts = 0

        while patches_found < max_patches_per_image and attempts < 3:
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
                background_patches.append(patch)
                patches_found += 1

            attempts += 1

    return background_patches

# Генератор загрузки
def create_dataloader(patches, label, batch_size=512):
    labels = np.full(len(patches), label)

    # Изменение размера
    transformed_patches = []

    for patch in patches:
        patch_image = transform(patch)
        transformed_patches.append(patch_image.numpy())

    tensor_X = torch.tensor(transformed_patches).float()
    tensor_y = torch.tensor(labels).long()
    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    yield dataloader

# Извлечение признаков из ResNet
def extract_resnet_features(dataloaders, resnet):
    resnet.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for dataloader in dataloaders:
            for images, labels in dataloader:
                features = resnet(images)

                all_features.extend(features.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    return np.array(all_features), np.array(all_labels)

# Главная функция вызова процессинга
def process_data(data_yaml_path, images_path, labels_path, image_files,
                 label_files, class_name='person', resnet=resnet_model):
    # Список классов
    class_names = load_classes(data_yaml_path)
    # Средние размеры целевого бибокса
    avg_width, avg_height = get_average_bbox_size(image_files, labels_path, class_names, class_name)
    # Извлечение патчей
    persons_patches = extract_persons(images_path, image_files, labels_path,
                                      avg_width, avg_height, class_names, class_name)
    background_patches = extract_background(images_path, image_files,
                                            labels_path, avg_width, avg_height, class_names, class_name, max_patches_per_image=3)
    # Даталоадеры
    persons_dataloader = create_dataloader(persons_patches, label=1)
    background_dataloader = create_dataloader(background_patches, label=0)
    # Выгрузка векторов
    X_person_features, y_person_labels = extract_resnet_features(persons_dataloader, resnet=resnet)
    X_background_features, y_background_labels = extract_resnet_features(background_dataloader, resnet=resnet)
    X = np.vstack((X_person_features, X_background_features))
    y = np.hstack((y_person_labels, y_background_labels))
    X, y = shuffle(X, y, random_state=42)

    return X, y