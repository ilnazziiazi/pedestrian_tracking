import os
import cv2
from torchvision.transforms import v2
import skimage as ski
from PIL import Image
import albumentations as A

def save_image(image, image_name, save_dir, extension="jpg"):
    os.makedirs(save_dir, exist_ok=True)
    img_save_path = os.path.join(save_dir, f"{image_name}.{extension}")
    cv2.imwrite(img_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Изображение сохранено: {img_save_path}")

def save_bboxes(bboxes, labels, image_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    label_file = os.path.join(save_dir, f"{image_name}.txt")
    
    with open(label_file, "w") as file:
        for bbox, label in zip(bboxes, labels):
            x_center, y_center, box_width, box_height = bbox
            file.write(f"{label} {x_center} {y_center} {box_width} {box_height}\n")
    print(f"Bounding boxes сохранены: {label_file}")

bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'])

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5)
], bbox_params=bbox_params)


def augment_image_with_bbox(image, bboxes, labels):
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=labels)
    return augmented['image'], augmented['bboxes'], augmented['class_labels']


# TorchVision аугментации
cropper = v2.RandomCrop(size=(200, 200))
jitter = v2.ColorJitter(brightness=0.5, hue=0.3)
hflipper = v2.RandomHorizontalFlip(p=1)
gray_transform = v2.Grayscale()

def random_crop_augment(img, name, save_dir):
    crops = [cropper(img) for _ in range(4)]
    for i, x in enumerate(crops):
        save_image(x, f"{name}_crops_{i}", save_dir)


def color_jitter_augment(img, name, save_dir):
    jittered_imgs = [jitter(img) for _ in range(3)]
    for i, x in enumerate(jittered_imgs):
        save_image(x, f"{name}_jitter_{i}", save_dir)


def grayscale_augment(img, name, save_dir):
    gray_img = gray_transform(img)
    save_image(gray_img, f"{name}_gray", save_dir)


def horizontal_flip_augment(img, name, save_dir):
    transformed_imgs = [hflipper(img)]
    for i, x in enumerate(transformed_imgs):
        save_image(x, f"{name}_hflipped_{i}", save_dir)


def add_gaussian_noise(img, name, save_dir):
    img_to_noise = ski.util.img_as_float64(img)
    noise_aug = ski.util.random_noise(img_to_noise, mode='gaussian', mean=0.01, var=0.0125)
    noise_aug = ski.util.img_as_ubyte(noise_aug)
    noise_aug = Image.fromarray(noise_aug)
    save_image(noise_aug, f"{name}_noise", save_dir)