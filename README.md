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

## Service demonstation with model inference

[![Project demo](https://img.youtube.com/vi/38uSmBjsTpY/0.jpg)](https://www.youtube.com/watch?v=38uSmBjsTpY)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
