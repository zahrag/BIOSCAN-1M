import os
import sys
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

from transformers import DetrFeatureExtractor
from crop_tool_sup.util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox
from crop_tool_sup.scripts.crop_images import expand_image
from crop_tool_sup.model.detr import load_model_from_ckpt
from resize_image import make_resize
from utils import write_in_hdf5, read_from_hdf5, read_tsv


class CustomArg:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def save_cropped_image(configs, img, cropped_img, chunk_number):
    """
    This function saves the cropped image in the corresponding file format of the dataset if the path is preset.
    :param configs: Configurations.
    :param img: Original image file.
    :param cropped_img: cropped image data array.
    :param chunk_number: chunk number to locate the image if using bioscan_1M_insect dataset structure.
    :return:
    """

    if configs['cropped_image_path'] is not None:
        if configs['data_structure'] == 'bioscan_1M_insect' and chunk_number is not None:
            cropped_img.save(os.path.join(configs['cropped_image_path'], f"part{chunk_number}/{os.path.basename(img)}"))
        else:
            cropped_img.save(os.path.join(configs['cropped_image_path'], os.path.basename(img)))

    if configs['cropped_hdf5_path'] is not None:
        if not os.path.isfile(configs['cropped_hdf5_path']):
            output_hdf5 = h5py.File(configs['cropped_hdf5_path'], 'w')
        else:
            output_hdf5 = h5py.File(configs['cropped_hdf5_path'], 'a')

        write_in_hdf5(output_hdf5, cropped_img, os.path.basename(img), image_dir=None, save_binary=False)


def cropping(args, image, model, feature_extractor):
    """
    :param args: non-dictionary configurations.
    :param image: image to crop.
    :param model: Model used for cropping.
    :param feature_extractor: Model used to extract features.
    :return: Cropped image.
    """

    encoding = feature_extractor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0)
    outputs = model(pixel_values=pixel_values, pixel_mask=None)
    bbox = get_bbox_from_output(outputs, image).detach().numpy()
    bbox = np.round(bbox, 0)
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    left, top, right, bottom = scale_bbox(args, left, top, right, bottom)
    args.background_color_R = 240
    args.background_color_G = 240
    args.background_color_B = 240
    if left < 0:
        border_size = 0 - left
        right = right - left
        left = 0
        image = expand_image(args, image, border_size, 'left')
    if top < 0:
        border_size = 0 - top
        bottom = bottom - top
        top = 0
        image = expand_image(args, image, border_size, 'top')
    if right > image.size[0]:
        border_size = right - image.size[0] + 1
        image = expand_image(args, image, border_size, 'right')
    if bottom > image.size[1]:
        border_size = bottom - image.size[1] + 1
        image = expand_image(args, image, border_size, 'bottom')
    cropped_img = image.crop((left, top, right, bottom))

    return cropped_img


def crop_image(configs, original_images, chunk_ids):
    """
    This function does the cropping.
    :param configs: Configurations.
    :param original_images: path list of uncropped images.
    :return:
    """

    if len(original_images) == 0:
        print("No images detected for cropping!")
        return

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    # Get non-dictionary type of the configurations used in the cropping tool.
    args = CustomArg(configs)
    model = load_model_from_ckpt(args)

    for orig_img in tqdm(original_images):
        if configs['data_format'] == "folder":
            if os.path.isfile(orig_img):
                try:
                    image = Image.open(orig_img)
                except:
                    print("Image not found in: " + orig_img)
                    exit(1)

        elif configs['data_format'] == "hdf5":
            input_hdf5 = h5py.File(configs['hdf5_path'], 'r')
            if configs['dataset_name'] in input_hdf5.keys():
                input_hdf5 = input_hdf5[configs['dataset_name']]
            if os.path.basename(orig_img) not in input_hdf5.keys():
                print("Image not found in: " + configs['hdf5_path'])
                exit(1)
            image = read_from_hdf5(input_hdf5, os.path.basename(orig_img), saved_as_binary=True)

        else:
            sys.exit("Wrong data_format: " + configs['data_format'] + " does not exist.")

        # Cropping the image
        cropped_img = cropping(args, image, model, feature_extractor)

        # Save the cropped image
        chunk_number = None
        if chunk_ids:
            chunk_number = chunk_ids[original_images.index(orig_img)]
        save_cropped_image(configs, orig_img, cropped_img, chunk_number)


def detect_uncropped_images(metadata, dataset_name,
                            image_path, cropped_image_path, hdf5_path, cropped_hdf5_path, read_format,
                            use_metadata=False, get_list=False):
    """
    Using metadata file, this function outputs a list of path to images, we want to crop.
    :param metadata: Path to the metadata file.
    :param dataset_name: Name of the dataset.
    :param image_path: Path to the directory where uncropped images are saved.
    :param cropped_image_path: Path to the directory where cropped images will be saved.
    :param hdf5_path: Path to the hdf5 file, which contains uncropped images.
    :param cropped_hdf5_path: Path to the hdf5 file where cropped images will be saved.
    :param read_format: File format to read data from.
    :param use_metadata: If using metadata for cropping?
    :param get_list: If get image path list?
    :return:
    """

    if not get_list:
        return

    list_of_uncropped_image_path, chunk_ids = [], []

    # Get the list of image names
    if use_metadata:
        df = read_tsv(metadata)
        image_names = df['image_file'].to_list()
        chunk_ids = df['chunk_number'].to_list()

    else:
        if read_format == "folder":
            image_names = os.listdir(image_path)

        elif read_format == "hdf5":
            file = h5py.File(hdf5_path, 'a')
            if dataset_name in file.keys():
                image_names = file[dataset_name].keys()
            else:
                image_names = file.keys()
        else:
            sys.exit("Wrong data_format: " + read_format + " does not exist.")

    # Detect uncropped images from image names list
    if read_format == "folder":
        pbar = tqdm(image_names)
        for img in pbar:
            pbar.set_description("Detect uncropped images")
            curr_image_dir = os.path.join(image_path, img)
            if not os.path.isfile(curr_image_dir):
                sys.exit(curr_image_dir + " does not exit in original image path.")

            curr_cropped_image_dir = os.path.join(cropped_image_path, img)
            curr_cropped_image_dir_ = os.path.join(image_path, "CROPPED_" + img)
            if not os.path.isfile(curr_cropped_image_dir) and not os.path.isfile(curr_cropped_image_dir_):
                list_of_uncropped_image_path.append(curr_image_dir)

    elif read_format == "hdf5":
        file = h5py.File(hdf5_path, 'a')
        cropped_file = h5py.File(cropped_hdf5_path, 'a')
        keys = file[dataset_name].keys()
        if dataset_name in cropped_file.keys():
            cropped_keys = cropped_file[dataset_name].keys()
        else:
            cropped_keys = cropped_file.keys()

        pbar = tqdm(image_names)
        for img in pbar:
            pbar.set_description("Detect uncropped images")
            if img not in keys:
                sys.exit(img + "does not exit in original hdf5 file.")

            if img not in cropped_keys and "CROPPED_" + img not in keys:
                list_of_uncropped_image_path.append(img)
    else:
        sys.exit("Wrong data_format: " + read_format + " does not exist.")

    return list_of_uncropped_image_path, chunk_ids


def run_crop_tool(configs):

    """
    This function runs as offline preprocessing to detect the images without a cropped version in the dataset, and to
    create a cropped version and save it in the corresponding data format.
    :param configs: Configurations.
    :return:
    """

    if not configs['crop_image']:
        return

    # Detect uncropped images
    path_to_uncropped_images, chunk_ids = detect_uncropped_images(configs['metadata_path'], configs['dataset_name'],
                                                                  configs['image_path'], configs['cropped_image_path'],
                                                                  configs['hdf5_path'], configs['cropped_hdf5_path'],
                                                                  configs['data_format'],
                                                                  use_metadata=configs['use_metadata'],
                                                                  get_list=True)

    # Crop original images without a cropped version and save in dataset file.
    crop_image(configs, path_to_uncropped_images, chunk_ids)

    # Resize cropped images to 256 on their smaller dimension
    make_resize(configs['cropped_image_path'],
                configs['resized_cropped_image_path'], configs['resized_cropped_hdf5_path'],
                save_binary=True, resize_dimension=256, zip_name="cropped_256.zip", resize_images=True)


