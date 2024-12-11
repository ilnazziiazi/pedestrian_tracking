import os
import random
from roboflow import Roboflow
from PIL import Image
from helpers1 import plot

import skimage
import skimage.io
import selective_search

def proceed_selective_search(img_sample_slice, n, shared_dict):
    print(f"Process {n} has started execution")
    dict_of_boxes = {}
    for img_path in img_sample_slice:
        img = skimage.io.imread(rf"{img_path}")
        boxes = selective_search.selective_search(img, mode='single', random_sort=False)
        boxes_filter = selective_search.box_filter(boxes, min_size=1, topN=80)
        dict_of_boxes[img_path] = boxes_filter
    shared_dict[n] = dict_of_boxes
    print(f"Process {n} has ended execution")