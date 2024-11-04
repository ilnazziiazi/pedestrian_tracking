import os
import cv2
import torch
from torchvision.transforms import v2
from PIL import Image
import skimage as ski
from roboflow import Roboflow

rf = Roboflow(api_key="iOJNkzHKPzOLoZ0dIkM9")
project = rf.workspace("aerial-person-detection").project("aerial-person-detection")
version = project.version(3)
version.download("yolov11", location='./data')

directory = "/Users/yanyshev_dima/Documents/Projects/AI Masters/Year Project/data/train/images/"
os.chdir(directory)

torch.manual_seed(0)

def save_image(image, image_name):
    img_save_dir = os.path.join(directory, image_name)
    img_save_dir += '.jpeg'
    cv2.imwrite(img_save_dir, image)




cropper = v2.RandomCrop(size=(200, 200))
jitter = v2.ColorJitter(brightness=.5, hue=.3)
hflipper = v2.RandomHorizontalFlip(p=1)

for file in os.listdir(directory):
    name = file.split(".")[0]
    img = Image.open(file)

    (resized_100, resized_250) = [v2.Resize(size=size)(img) for size in (100, 250)]
    [save_image(x, name + 'resized' + str(i)) for i, x in enumerate((resized_100, resized_250))]

    (top_left, top_right, bottom_left, bottom_right, center) = v2.FiveCrop(size=(300, 300))(img)
    [save_image(x, name + "five_crop" + str(i)) for i, x in
     enumerate((top_left, top_right, bottom_left, bottom_right, center))]

    crops = [cropper(img) for _ in range(4)]
    [save_image(x, name + 'crops' + str(i)) for i, x in enumerate(crops)]

    gray_img = v2.Grayscale()(img)
    save_image(gray_img, name + "gray")

    jittered_imgs = [jitter(img) for _ in range(3)]
    [save_image(x, name + 'jitter' + str(i)) for i, x in enumerate(jittered_imgs)]

    transformed_imgs = [hflipper(img) for _ in range(1)]
    [save_image(x, name + 'hflipped' + str(i)) for i, x in enumerate(transformed_imgs)]

    img_to_noise = ski.util.img_as_float64(img)
    noise_aug = ski.util.random_noise(img_to_noise, mode='gaussian', mean=0.01, var=.0125)
    noise_aug = ski.util.img_as_ubyte(noise_aug)
    noise_aug = Image.fromarray(noise_aug)
    save_image(noise_aug, name + "noise")

    print(file)
    print('Done')