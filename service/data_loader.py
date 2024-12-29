import os
from typing import Tuple, List, Any
import yaml
import zipfile
from pathlib import Path
from fastapi import UploadFile, HTTPException

# Сохранение и распаковка данных
def save_and_unpack(archive: UploadFile, output_dir: str) -> tuple[Path | None, Path | None, list[Any], list[Any]]:
    temp_dir = Path(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    archive_path = temp_dir / archive.filename
    with archive_path.open("wb") as f:
        f.write(archive.file.read())

    if not zipfile.is_zipfile(archive_path):
        raise HTTPException(status_code=400, detail="Файл должен быть zip-архивом")

    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    images_path = None
    labels_path = None
    data_yaml_path = None

    # Извлекаем yaml-файл с классами
    for root, dirs, files in os.walk(temp_dir):
        if "data.yaml" in files:
            data_yaml_path = Path(root) / "data.yaml"
            break

    if not data_yaml_path:
        raise HTTPException(
            status_code=400,
            detail="YAML-файл с классами не найден в архиве"
        )

    # Извлекаем папки с изображениями и боксами
    for root, dirs, _ in os.walk(temp_dir):
        if "images" in dirs and "labels" in dirs:
            images_path = Path(root) / "images"
            labels_path = Path(root) / "labels"
            break

    if not images_path:
        raise HTTPException(
            status_code=400,
            detail="Папка images не найдена в архиве"
        )

    if not labels_path:
        raise HTTPException(
            status_code=400,
            detail="Папка labels не найдена в архиве"
        )

    image_files = [file.stem for file in images_path.glob("*.jpg")]
    label_files = [file.stem for file in labels_path.glob("*.txt")]

    if not image_files or not label_files:
        raise HTTPException(
            status_code=400,
            detail="Папки images и labels должны содержать файлы с расширением .jpg и .txt"
        )

    return data_yaml_path, images_path, labels_path, image_files, label_files

# Загрузка классов
def load_classes(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# Загрузка boxes
def load_bounding_boxes(image_name, labels_dir):
    label_file = labels_dir / f'{image_name}.txt'

    if not label_file.exists():
        print(f"Файл меток отсутствует: {label_file}")
        return []

    boxes = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])  # Идентификатор класса
            x_center, y_center, box_width, box_height = map(float, parts[1:5])
            boxes.append((class_id, x_center, y_center, box_width, box_height))
    return boxes