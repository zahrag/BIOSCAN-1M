import os
from utils import resize_image, make_directory


def make_resize(configs):

    if configs["resized_image_path"] is None:
        print("No path is set to the directory where the resized images will be saved!")
        return

    full_size_img_path = configs["image_path"]
    resize_img_path = configs["resized_image_path"]
    for img in os.listdir(full_size_img_path):
        resize_image(f"{full_size_img_path}/{img}", f"{resize_img_path}/{img}", resize_dimension=256)
