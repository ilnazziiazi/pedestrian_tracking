"""
Main file for training Yolo model on Pascal VOC dataset

"""

# Основные импорты библиотек
import torch  # Основная библиотека для глубокого обучения
import torchvision.transforms as transforms  # Модуль для преобразования изображений
import torch.optim as optim  # Оптимизаторы для обучения
from tqdm import tqdm  # Прогресс-бар для отслеживания обучения
from torch.utils.data import DataLoader  # Загрузчик данных
from model import Yolov1  # Наша модель YOLO
from dataset import VOCDataset  # Датасет Pascal VOC
import os
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
from datetime import datetime  # Добавим для timestamp в логах

# Настройка случайного сида для воспроизводимости результатов
seed = 123
torch.manual_seed(seed)

# Гиперпараметры модели
LEARNING_RATE = 2e-5  # Скорость обучения - насколько большими шагами модель обучается
DEVICE = "cuda" if torch.cuda.is_available else "cpu"  # Используем GPU если доступен, иначе CPU
BATCH_SIZE = 16  # Количество изображений, обрабатываемых за раз
WEIGHT_DECAY = 0
EPOCHS = 2  # Количество полных проходов по датасету
NUM_WORKERS = 12  # Количество параллельных процессов для загрузки данных
PIN_MEMORY = True  # Ускорение передачи данных на GPU
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/train/images"
LABEL_DIR = "data/train/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        # Применяем последовательно все преобразования к изображению
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training iteration")
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)  # Перемещаем данные на GPU/CPU
        out = model(x)  # Прямой проход через модель
        loss = loss_fn(out, y)  # Вычисляем функцию потерь
        mean_loss.append(loss.item())  # Сохраняем значение потерь
        
        # Обратное распространение ошибки
        optimizer.zero_grad()  # Обнуляем градиенты
        loss.backward()  # Вычисляем градиенты
        optimizer.step()  # Обновляем веса модели

        # Более информативный постфикс для progress bar
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'batch': f'{batch_idx}/{len(train_loader)}',
            'avg_loss': f'{sum(mean_loss)/len(mean_loss):.4f}'
        })

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch training completed. Mean loss: {sum(mean_loss)/len(mean_loss):.4f}")


def main():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting YOLO training with following parameters:")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    
    model = Yolov1(split_size=7, num_boxes=2, num_classes=12).to(DEVICE)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model initialized")
    
    # Инициализируем оптимизатор Adam
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    best_map = 0.0  # Отслеживаем лучшее значение mAP

    if LOAD_MODEL:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading checkpoint from {LOAD_MODEL_FILE}")
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # Add error checking for CSV files
    train_csv = "/home/ilnazzia/Git/team_49_pedestrian_tracking/yolo_from_scratch/aerial/train.csv"
    test_csv = "/home/ilnazzia/Git/team_49_pedestrian_tracking/yolo_from_scratch/aerial/test.csv"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets...")
    train_dataset = VOCDataset(
        train_csv,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    test_dataset = VOCDataset(
        test_csv, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Datasets loaded. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,  # Перемешиваем данные
        drop_last=True,  # Отбрасываем неполный последний батч
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training loop")
    for epoch in range(EPOCHS):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch [{epoch+1}/{EPOCHS}]")
        print("Getting bounding boxes for mAP calculation...")
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train mAP: {mean_avg_prec:.4f}")

        if mean_avg_prec > best_map:
            best_map = mean_avg_prec
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New best mAP! Saving checkpoint...")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch{epoch+1}_map{mean_avg_prec:.4f}.pth.tar")

        train_fn(train_loader, model, optimizer, loss_fn)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training completed! Best mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()