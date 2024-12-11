import os
import csv

# Получаем абсолютный путь к текущей директории файла
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, "train")
train_images_path = os.path.join(train_path, "images")
train_labels_path = os.path.join(train_path, "labels")
train_files = [f for f in os.listdir(train_images_path) if f.endswith('.jpg')]

with open("train.csv", mode="w", newline="") as train_file:
    writer = csv.writer(train_file)
    for image_file in train_files:
        image_path = os.path.join(train_images_path, image_file)
        text_file = image_file.replace(".jpg", ".txt")
        text_path = os.path.join(train_labels_path, text_file)
        data = [image_path, text_path]
        writer.writerow(data)

test_path = os.path.join(current_dir, "valid")
test_images_path = os.path.join(test_path, "images")
test_labels_path = os.path.join(test_path, "labels")
test_files = [f for f in os.listdir(test_images_path) if f.endswith('.jpg')]

with open("test.csv", mode="w", newline="") as test_file:
    writer = csv.writer(test_file)
    for image_file in test_files:
        image_path = os.path.join(test_images_path, image_file)
        text_file = image_file.replace(".jpg", ".txt")
        text_path = os.path.join(test_labels_path, text_file)
        data = [image_path, text_path]
        writer.writerow(data)
