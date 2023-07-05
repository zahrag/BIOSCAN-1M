import os
from utils import resize_image


def make_resize(configs):

    if not configs["resize_image"]:
        return

    root = ""
    full_size_img_path = f"{root}/full_size_images/img.jpg"
    resize_img_path = f"{root}/resize_images/img.jpg"
    for img in os.listdir(full_size_img_path):
        resize_image(f"{full_size_img_path}/{img}", f"{resize_img_path}/{img}", resize_dimension=256)
