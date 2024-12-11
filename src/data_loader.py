import yaml
import shutil
import os
from pathlib import Path
from roboflow import Roboflow


def load_classes(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

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

def download_dataset(api_key, workspace, project_name, version_num, dest_dir):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download("yolov11")

    current_dir = Path.cwd()
    destination_dir = Path(dest_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    potential_folders = list(current_dir.glob("Aerial-Person-Detection*"))

    if not potential_folders:
        print("Папка с префиксом 'Aerial-Person-Detection' не найдена после загрузки.")
        return None

    source_dir = potential_folders[0]
    target_dir = destination_dir / source_dir.name

    if target_dir.exists():
        print(f"Папка {target_dir} уже существует. Выполняется удаление...")
        shutil.rmtree(target_dir)

    try:
        shutil.move(str(source_dir), str(target_dir))
        print(f"Датасет успешно перемещён в {target_dir}")
    except Exception as e:
        print(f"Ошибка при перемещении датасета: {e}")
        return None

    return target_dir