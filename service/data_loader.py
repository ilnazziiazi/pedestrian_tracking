import os
from typing import Tuple, List, Dict
import yaml
import zipfile
from pathlib import Path
from fastapi import UploadFile, HTTPException
from collections import defaultdict
import asyncio
import aiofiles


# Cохранение и распаковка данных
async def save_and_unpack(archive: UploadFile, output_dir: str) -> Tuple[Path, Path, Path]:
    temp_dir = Path(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    archive_path = temp_dir / archive.filename
    with archive_path.open("wb") as f:
        f.write(archive.file.read())

    if not zipfile.is_zipfile(archive_path):
        raise HTTPException(status_code=400,
                            detail="Файл должен быть zip-архивом")

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
        raise HTTPException(
            status_code=400,
            detail="YAML-файл с классами не найден в архиве."
        )
    if not images_path or not labels_path:
        raise HTTPException(
            status_code=400,
            detail="Папки images и labels не найдены в архиве."
        )

    # Проверяем расширения
    if not any(images_path.glob("*.jpg")):
        raise HTTPException(
            status_code=400,
            detail="Папка images не содержит файлов с расширением .jpg."
        )
    if not any(labels_path.glob("*.txt")):
        raise HTTPException(
            status_code=400,
            detail="Папка labels не содержит файлов с расширением .txt."
        )

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
        raise FileNotFoundError(f"Файл отсутствует: {label_file}.")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_boxes, label_file)


# Получение сохраненных при распаковке файлов
def get_files(path, extension):
    return [file.stem for file in path.glob(f"*.{extension}")]


# Поиск классов в файле
async def process_label(label_file: Path, class_names: List[str],  person_class_id: int,
                        images_with_person: set, class_bbox_count: Dict[str, int],
                        class_image_count: Dict[str, int]):
    classes_in_image = set()
    try:
        if label_file.exists():
            async with aiofiles.open(label_file, "r") as file:
                async for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    class_name = class_names[class_id]
                    class_bbox_count[class_name] += 1

                    if class_id == person_class_id:
                        images_with_person.add(label_file.stem)

                    if class_name not in classes_in_image:
                        class_image_count[class_name] += 1
                        classes_in_image.add(class_name)

    except (ValueError, IndexError, FileNotFoundError):
        pass
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Произошла ошибка: {str(e)}."
        )


# Загрузка агрегированной статистики по входным данным
async def load_image_groups(images_path: Path, labels_path: Path,
                            data_yaml_path: Path) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    try:
        images_with_person = set()
        class_bbox_count = defaultdict(int)
        class_image_count = defaultdict(int)

        image_files = get_files(images_path, "jpg")
        class_names = await load_classes(data_yaml_path)
        person_class_id = class_names.index("person")

        # Вызов параллельного поиска
        tasks = [
            process_label(labels_path / f"{image_name}.txt", class_names, person_class_id,
                          images_with_person, class_bbox_count, class_image_count)
            for image_name in image_files
        ]

        await asyncio.gather(*tasks)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Произошла ошибка: {str(e)}."
        )

    return list(images_with_person), class_bbox_count, class_image_count
