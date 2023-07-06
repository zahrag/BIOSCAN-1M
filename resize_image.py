import os
from utils import resize_image, make_directory
import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError


def make_resize(configs, saved_as_binary_data=True, resize_dimension=256):

    if configs["resized_image_path"] is not None:
        full_size_img_path = configs["image_path"]
        resized_img_path = configs["resized_image_path"]
        for img in os.listdir(full_size_img_path):
            resize_image(f"{full_size_img_path}/{img}", f"{resized_img_path}/{img}", resize_dimension=resize_dimension)

        if configs['resized_hdf5_path'] is not None:

            with h5py.File(configs['resized_hdf5_path'], 'w') as hdf5:
                for img in os.listdir(resized_img_path):
                    image_dir = f"{resized_img_path}/{img}"

                    if saved_as_binary_data:
                        with open(image_dir, 'rb') as img_f:
                            binary_data = img_f.read()
                        binary_data_np = np.asarray(binary_data)
                        hdf5.create_dataset(f'{img}', data=binary_data_np)
                    else:

                        try:
                            image = Image.open(image_dir)
                            image_array = np.array(image)
                            hdf5.create_dataset(f'{img}', data=image_array)

                        except UnidentifiedImageError:
                            print(f"{img} Corrupted.")
                            continue
