import os
from typing import Tuple, List
import yaml
import zipfile
from pathlib import Path
from fastapi import UploadFile, HTTPException
import asyncio

# Cохранение и распаковка данных
async def save_and_unpack(archive: UploadFile, output_dir: str) -> Tuple[Path, Path, Path]:
    temp_dir = Path(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    archive_path = temp_dir / archive.filename
    with archive_path.open("wb") as f:
        f.write(archive.file.read())

    if not zipfile.is_zipfile(archive_path):
        raise HTTPException(status_code=400, detail="Файл должен быть zip-архивом")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, zipfile.ZipFile(archive_path, 'r').extractall, temp_dir)

    images_path = None
    labels_path = None
    data_yaml_path = None

    # Поиск data.yaml и папок images и labels
    for root, dirs, files in os.walk(temp_dir):
        if "data.yaml" in files:
            data_yaml_path = Path(root) / "data.yaml"
        if "images" in dirs and "labels" in dirs:
            images_path = Path(root) / "images"
            labels_path = Path(root) / "labels"
            if data_yaml_path and images_path and labels_path:
                break

    if not data_yaml_path:
        raise HTTPException(status_code=400, detail="YAML-файл с классами не найден в архиве")
    if not images_path or not labels_path:
        raise HTTPException(status_code=400, detail="Папки images и labels не найдены в архиве")

    # Проверяем расширения
    if not any(images_path.glob("*.jpg")):
        raise HTTPException(status_code=400, detail="Папка images не содержит файлов с расширением .jpg")
    if not any(labels_path.glob("*.txt")):
        raise HTTPException(status_code=400, detail="Папка labels не содержит файлов с расширением .txt")

    return data_yaml_path, images_path, labels_path

# Загрузка классов
async def load_classes(data_yaml_path: Path) -> List[str]:
    loop = asyncio.get_event_loop()
    with open(data_yaml_path, 'r') as f:
        data = await loop.run_in_executor(None, yaml.safe_load, f)
    return data['names']

# Загрузка bounding boxes
def read_boxes(label_file):
    boxes = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:5])
            boxes.append((class_id, x_center, y_center, box_width, box_height))
    return boxes

async def load_bounding_boxes(image_name: str, labels_dir: Path) -> List[Tuple[int, float, float, float, float]]:
    label_file = labels_dir / f"{image_name}.txt"

    if not label_file.exists():
        print(f"Файл отсутствует: {label_file}")
        return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_boxes, label_file)