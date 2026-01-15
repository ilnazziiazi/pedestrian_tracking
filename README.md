# Pedestrian Tracking
## Project Overview

A service was created to identify pedestrians in videos, calculate pedestrian traffic, its average speed and density distribution throughout the day, as well as pedestrians by gender, age, and wealth. Such analytics can be useful for understanding the target audience for companies placing advertisements on the street.

## Documentation
- [Technical documentation](./report.md)
- [EDA results](./EDA.md)

## To run, you need to:
1. Add a .env file to the project root.
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to work with the GPU.
3. Add the model [weights](https://drive.google.com/file/d/1L8H0u8CHvF3oKpP4jApN2guUblrAq0sW/view?usp=share_link) to the `models/yolov11.pt` folder.
4. Run `docker compose up --build`.


## Experiments
### Detection
| Model | Augmentation | Parameters | mAP loU 0.50 | mAP loU 0.50-0.95 | Training time (RTX 3090) |
|---|---|---|---|---|---|
|yolo12| crop, mosaic| Imgsz 960, epoch 46, batch 4| 0.597| 0.380| ~8 hours|
|yolo12| crop, mosaic | imgsz 640, epoch 100, batch 12 | 0.532 | 0.333 | ~8 hours | 
| yolo12 | default | imgsz 640, epoch 100, batch 12 | 0.516 | 0.322 | ~7 hours |
| detectron2 + faster rcnn 101 32x8d FPN | default | LR 0.00025, max_iter 1000, batch 4 | 0.402 | 0.089 | ~30 minutes |
| Deformable-DETR | default | Ir 0.0001, epoch 10, batch 2 | 0.154 | 0.065 | ~5 hours |

We chose the yolo12 with Imgsz 960 model for further experiments due to its high accuracy.
![](content/image01.gif)
### Tracking
| Method | HOTA | IDF1 | MOTA | MOTP |
|---|---|---|---|---|
| ByteTrack | 0.34| 0.44| 0.46| 0.72 |
| DeepSort| 0.16| 0.17| 0.10| 0.71 |

ByteTrack showed better results in all metrics, so we used it for further experiments

![](content/image02.gif)

## Service demonstation with model inference

[![Project demo](https://img.youtube.com/vi/38uSmBjsTpY/0.jpg)](https://www.youtube.com/watch?v=38uSmBjsTpY)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
