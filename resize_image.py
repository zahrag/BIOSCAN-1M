import os
from utils import resize_image, make_directory
import h5py
import numpy as np


def make_resize(configs):

    if configs["resized_image_path"] is not None:
        full_size_img_path = configs["image_path"]
        resized_img_path = configs["resized_image_path"]
        for img in os.listdir(full_size_img_path):
            resize_image(f"{full_size_img_path}/{img}", f"{resized_img_path}/{img}", resize_dimension=256)

        if configs['resized_hdf5_path'] is not None:
            with h5py.File(configs['resized_hdf5_path'], 'w') as hdf5:
                for img in os.listdir(resized_img_path):
                    with open(f"{resized_img_path}/{img}", 'rb') as img_f:
                        binary_data = img_f.read()
                    binary_data_np = np.asarray(binary_data)
                    hdf5.create_dataset(f'{img}', data=binary_data_np)
