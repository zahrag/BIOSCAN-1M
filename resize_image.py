import os
from utils import resize_image, create_zip, write_in_hdf5, remove_file
import h5py
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def make_resize(full_size_img_path, resized_img_path, resized_hdf5_path,
                save_binary=True, resize_dimension=256, zip_name=None, resize_images=False):
    """
    This function resizes images to 256 on their smaller dimension, and saves both in folder and hdf5 if
    the path to these are preset.
    :param full_size_img_path: Path to the directory of the full sized images .
    :param resized_img_path: Path to the directory to save resized images.
    :param resized_hdf5_path: Path to the hdf5 file to save resized images.
    :param save_binary: if True, less space required.
    :param resize_dimension: Dimension to resize images.
    :param zip_name: If not None, the resized image folder is compressed to <zip_name>.zip
    :param resize_images: If resize images?
    :return:
    """

    if not resize_images:
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
            # remove_file(img, full_size_img_path)
            print(f"{img} Corrupted.")
            continue

    if zip_name is not None:
        create_zip(source_folder=resized_img_path, output_zip=f"{os.path.dirname(resized_img_path)}/{zip_name}")

    if resized_hdf5_path is not None:
        if os.path.isfile(resized_hdf5_path):
            hdf5 = h5py.File(resized_hdf5_path, 'a')
        else:
            hdf5 = h5py.File(resized_hdf5_path, 'w')

        pbar = tqdm(os.listdir(resized_img_path))
        for image_file in pbar:
            if image_file in hdf5.keys():
                print(f"{image_file} Exists: Skip!")
                continue
            pbar.set_description(f"Archive resized image on a HDF5 file in:\n{resized_hdf5_path}.")
            image_dir = f"{resized_img_path}/{image_file}"

            try:
                image = Image.open(image_dir)
                image.verify()
                write_in_hdf5(hdf5, image, image_file, image_dir=image_dir, save_binary=save_binary)

            except UnidentifiedImageError:
                # remove_file(image_file, resized_img_path)
                print(f"{image_file} Corrupted.")
                continue



