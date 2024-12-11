import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def display_image_with_boxes(image_name, images_dir, labels_dir, class_names, 
                             load_bounding_boxes=None, mode="real"):
    image_path = images_dir / f'{image_name}.jpg'
    img = Image.open(image_path)
    img_width, img_height = img.size

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    if mode == "real" and load_bounding_boxes is not None:
        boxes = load_bounding_boxes(image_name, labels_dir)
        rect_color = 'r'

        for (class_id, x_center, y_center, box_width, box_height) in boxes:
            x_min = (x_center - box_width / 2) * img_width
            y_min = (y_center - box_height / 2) * img_height
            width = box_width * img_width
            height = box_height * img_height

            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=rect_color, facecolor='none')
            ax.add_patch(rect)

            class_name = class_names[class_id]
            ax.text(x_min, y_min - 5, class_name, color=rect_color, fontsize=10, va='bottom', ha='left')

    elif mode == "pred" and load_bounding_boxes is not None:
        boxes = load_bounding_boxes(image_name, labels_dir)
        rect_color = 'g'

        for (class_id, x_center, y_center, box_width, box_height) in boxes:
            if class_id == 1:
                x_min = (x_center - box_width / 2) * img_width
                y_min = (y_center - box_height / 2) * img_height
                width = box_width * img_width
                height = box_height * img_height

                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=rect_color, facecolor='none')
                ax.add_patch(rect)

                ax.text(x_min, y_min - 5, "Person", color=rect_color, fontsize=10, va='bottom', ha='left')

    else:
        print("Указан неверный режим или отсутствуют необходимые данные.")

    plt.axis('off')
    plt.show()