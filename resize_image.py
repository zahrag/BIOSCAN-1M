import os
from utils import resize_image
import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def make_resize(full_size_img_path, resized_img_path, resized_hdf5_path, saved_as_binary_data=True, resize_dimension=256):
    """
    This function resizes images to 256 on their smaller dimension, and saves both in folder and hdf5 if
    the path to these are preset.
    :param full_size_img_path: Path to the full sized images.
    :param resized_img_path: Path to the directory to save resized images.
    :param resized_hdf5_path: Path to the hdf5 file to save resized images.
    :param saved_as_binary_data: if True, less space required.
    :param resize_dimension: Dimension to resize images.
    :return:
    """

    if full_size_img_path is None:
        print("No path is set to save resized images!")
        return

    pbar = tqdm(os.listdir(full_size_img_path))
    for img in pbar:
        resize_image(f"{full_size_img_path}/{img}", f"{resized_img_path}/{img}", resize_dimension=resize_dimension)

    if resized_hdf5_path is not None:
        with h5py.File(resized_hdf5_path, 'w') as hdf5:
            pbar = tqdm(os.listdir(resized_img_path))
            for img in pbar:
                image_dir = f"{resized_img_path}/{img}"
                try:
                    image = Image.open(image_dir)
                    image.verify()
                except UnidentifiedImageError:
                    print(f"{image} Corrupted.")
                    continue

                if saved_as_binary_data:
                    with open(image_dir, 'rb') as img_f:
                        binary_data = img_f.read()
                    binary_data_np = np.asarray(binary_data)
                    hdf5.create_dataset(f'{img}', data=binary_data_np)
                else:
                    image_array = np.array(image)
                    hdf5.create_dataset(f'{img}', data=image_array)

