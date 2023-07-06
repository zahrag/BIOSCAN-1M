import os
from utils import resize_image, create_zip
import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def make_resize(full_size_img_path, resized_img_path, resized_hdf5_path,
                saved_as_binary_data=True, resize_dimension=256, zip_name=None):
    """
    This function resizes images to 256 on their smaller dimension, and saves both in folder and hdf5 if
    the path to these are preset.
    :param full_size_img_path: Path to the full sized images.
    :param resized_img_path: Path to the directory to save resized images.
    :param resized_hdf5_path: Path to the hdf5 file to save resized images.
    :param saved_as_binary_data: if True, less space required.
    :param resize_dimension: Dimension to resize images.
    :param zip_name: If not None, the resized image folder is compressed to <zip_name>.zip
    :return:
    """

    if resized_img_path is None:
        print("No path is set to save resized images!")
        return

    pbar = tqdm(os.listdir(full_size_img_path))
    for img in pbar:
        if os.path.isfile(f"{resized_img_path}/{img}"):
            print(f"{img} Exists: Skip!")
            continue
        try:
            image = Image.open(f"{full_size_img_path}/{img}")
            image.verify()
            pbar.set_description(f"Resize image to {resize_dimension} on the smaller dimension.")
            resize_image(f"{full_size_img_path}/{img}", f"{resized_img_path}/{img}", resize_dimension=resize_dimension)
        except UnidentifiedImageError:
            # os.remove(f"{full_size_img_path}/{img}")
            print(f"{img} Corrupted.")
            continue

    if zip_name is not None:
        create_zip(source_folder=resized_img_path, output_zip=f"{os.path.dirname(resized_img_path)}/{zip_name}")

    if resized_hdf5_path is not None:
        if os.path.isfile(resized_hdf5_path):
            hdf5 = h5py.File(resized_hdf5_path, 'a')
        else:
            hdf5 = h5py.File(resized_hdf5_path, 'w')

        keys = hdf5.keys()
        pbar = tqdm(os.listdir(resized_img_path))
        for img in pbar:
            if img in keys:
                print(f"{img} Exists: Skip!")
                continue
            pbar.set_description(f"Archive resized image on a HDF5 file in:\n{resized_hdf5_path}.")
            image_dir = f"{resized_img_path}/{img}"

            try:
                image = Image.open(image_dir)
                image.verify()
                if saved_as_binary_data:
                    with open(image_dir, 'rb') as img_f:
                        binary_data = img_f.read()
                    binary_data_np = np.asarray(binary_data)
                    hdf5.create_dataset(f'{img}', data=binary_data_np)
                else:
                    image_array = np.array(image)
                    hdf5.create_dataset(f'{img}', data=image_array)

            except UnidentifiedImageError:
                # os.remove(image_dir)
                print(f"{img} Corrupted.")
                continue



