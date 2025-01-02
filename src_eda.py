import os
import shutil
from pathlib import Path

import yaml
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image

import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter


def get_image_size(image_names, specific_images=None):
    width_list, height_list, proportions_list, image_count = [], [], [], 0

    images_to_process = specific_images if specific_images is not None else image_names
    
    for image_file in images_to_process:
        if image_file in train_images:
            image_path = train_images_dir / f'{image_file}.jpg'
        else:
            image_path = valid_images_dir / f'{image_file}.jpg'
            
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                width_list.append(width)
                height_list.append(height)
                proportions_list.append(width / height)
                image_count += 1
        except Exception as e:
            print(f'Ошибка при обработке файла {image_file}: {e}')
            continue

    if image_count == 0:
        return 0, 0, 0

    img_size_stats = {
        'avg_width': round(np.array(width_list).mean()),
        'avg_height': round(np.array(height_list).mean()),
        'avg_proportion': round(np.array(proportions_list).mean(),2),
        'min_width': np.array(width_list).min(),
        'min_height': np.array(height_list).min(),
        'min_proportion': round(np.array(proportions_list).min(),2),
        'max_width': np.array(width_list).max(),
        'max_height': np.array(height_list).max(),
        'max_proportion': round(np.array(proportions_list).max(),2),
        'image_count': image_count
    }
    
    return img_size_stats, width_list, height_list, proportions_list


###
def get_grouped_hist(group_x_data, group_y_data, group_x_name, group_y_name, var_name, figsize=(7, 5), bins=10):
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(group_x_data, bins=bins, alpha=0.6, label=f'Image {var_name} {group_x_name}', color='blue', edgecolor='white', zorder=2)
    ax.hist(group_y_data, bins=bins, alpha=0.7, label=f'Image {var_name} {group_y_name}', color='forestgreen', edgecolor='white', zorder=2)
    ax.set_xlabel(f'Image {var_name}')
    ax.set_ylabel('Image Count')
    ax.set_title(f'Distribution of Image {var_name} by Groups')
    ax.legend().set_zorder(3)
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)

    file_name = f'{var_name}_distribution_grouped_by_{group_x_name}.jpg'
    
    plt.show()


def plot_distribution(data, metric_name):
    classes = list(data.keys())
    counts = list(data.values())
    
    plt.figure(figsize=(7, 5))
    plt.bar(classes, counts, color='blue', edgecolor='white')
    
    plt.xlabel('Class')
    plt.ylabel(f'{metric_name}')
    plt.title(f'Distribution of {metric_name} per Class')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.show()


def plot_hist(data, metric_name):
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=100, color='blue', edgecolor='white')
    
    plt.xlabel(f'{metric_name} per Image')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {metric_name} per Image')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.show()


### Heatmap
def get_bboxes_heatmap(image_names, image_shape=(500, 500), specific_classes=None):
    heatmap = np.zeros(image_shape, dtype=np.float32)

    for image_name in image_names:
        
        label_file = (labels_path if image_name in image_files else None) / f'{image_name}.txt'
        image_file = (images_path if image_name in image_files else None) / f'{image_name}.jpg'

        try:
            with Image.open(image_file) as img:
                img_width, img_height = img.size

            with open(label_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    class_name = class_names[class_id]
                    x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:5])

                    if specific_classes is not None and class_name not in specific_classes:
                        continue

                    x_center = x_center_norm * img_width
                    y_center = y_center_norm * img_height
                    box_width = width_norm * img_width
                    box_height = height_norm * img_height

                    xmin = int(x_center - box_width / 2)
                    ymin = int(y_center - box_height / 2)
                    xmax = int(x_center + box_width / 2)
                    ymax = int(y_center + box_height / 2)

                    x_start = int((xmin / img_width) * image_shape[1])
                    y_start = int((ymin / img_height) * image_shape[0])
                    x_end = int((xmax / img_width) * image_shape[1])
                    y_end = int((ymax / img_height) * image_shape[0])

                    heatmap[y_start:y_end, x_start:x_end] += 1

        except Exception as e:
            print(f'Ошибка при обработке файла {label_file}: {e}')
            continue

    return heatmap


def plot_hitmap(hitmap_matrix, metric_name):
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(hitmap_matrix, cmap='hot', cbar=True)
    plt.title(f'Heatmap of {metric_name} Position')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()