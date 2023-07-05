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


class CustomArg:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def save_cropped_image(configs, img, cropped_img, output_hdf5, file):
    """
    This function saves the cropped image in the corresponding file format of the dataset.
    :param configs: Configurations.
    :param img: Original image file.
    :param cropped_img: cropped image.
    :param output_hdf5: HDF5 output file if already exits.
    :param file: HDF5 file if just created.
    :return:
    """

    if configs['data_format'] == "":
        if configs['output_dir'] is not None:
            cropped_img.save(os.path.join(configs['output_dir'], "CROPPED_" + os.path.basename(img)))
        else:
            cropped_img.save(os.path.join(os.path.dirname(img), "CROPPED_" + os.path.basename(img)))

    if configs['data_format'] == "hdf5":
        if configs['output_dir'] is not None:
            output_hdf5.create_dataset("CROPPED_" + img, data=cropped_img)
        else:
            file[configs['dataset_name']].create_dataset("CROPPED_" + img, data=cropped_img)


def crop_image(configs, original_images):
    """
    This function does the cropping.
    :param configs: Configurations.
    :param original_images: path to the list of uncropped images.
    :return:
    """

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    # Get non-dictionary type of the configurations used in the cropped tool.
    args = CustomArg(configs)

    model = load_model_from_ckpt(args)
    output_hdf5 = None
    if configs['data_format'] == "hdf5":
        file = h5py.File(configs['image_hdf5'], 'a')
        if configs['output_hdf5'] is not None:
            if not os.path.isfile(configs['output_hdf5']):
                output_hdf5 = h5py.File(configs['output_hdf5'], 'w')
            else:
                output_hdf5 = h5py.File(configs['output_hdf5'], 'a')

    for orig_img in tqdm(original_images):
        if configs['data_format'] == "":
            file = None
            if os.path.isfile(orig_img):
                try:
                    image = Image.open(orig_img)
                except:
                    print("Image not found in: " + orig_img)
                    exit(1)

        elif configs['data_format'] == "hdf5":
            data = np.asarray(file[configs['dataset_name']][orig_img])
            image = Image.fromarray(data)

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

        # Save the cropped image
        save_cropped_image(configs, orig_img, cropped_img, output_hdf5, file)


def detect_uncropped_images(configs):

    """
    This function detects the images in the dataset file, which do not have a cropped version.
    :param configs: Configurations.
    :return: Path to the list of the uncropped images of the dataset
    """

    list_of_uncropped_image_path = []
    df = pd.read_csv(configs['meta_path'], sep='\t', low_memory=False)
    image_names = df['image_file'].to_list()

    if configs['data_format'] == "":
        pbar = tqdm(image_names)
        for img in pbar:
            pbar.set_description("Detect uncropped images")
            curr_image_dir = os.path.join(configs['image_path'], img)
            curr_cropped_image_dir = os.path.join(configs['output_dir'], "CROPPED_" + img)
            if not os.path.isfile(curr_image_dir):
                sys.exit(curr_image_dir + " does not exit")
            if not os.path.isfile(curr_cropped_image_dir):
                list_of_uncropped_image_path.append(curr_image_dir)
    elif configs['data_format'] == "hdf5":
        file = h5py.File(configs['image_hdf5'], 'a')
        output_file = h5py.File(configs['output_hdf5'], 'a')
        keys = file[configs['dataset_name']].keys()
        if configs['dataset_name'] in output_file.keys():
            output_keys = output_file[configs['dataset_name']].keys()
        else:
            output_keys = output_file.keys()
        pbar = tqdm(image_names)
        for i in pbar:
            pbar.set_description("Detect uncropped images")
            if i not in keys:
                sys.exit(i + "does not exit")
            if 'CROPPED_' + i not in output_keys:
                list_of_uncropped_image_path.append(i)
    else:
        sys.exit("Wrong data_format: " + configs['data_format'] + " does not exist.")

    return list_of_uncropped_image_path


def run_crop_tool(configs):

    """
    This function runs as offline preprocessing to detect the images without a cropped version in the dataset, and to
    create a cropped version and save it in the corresponding data format.
    :param configs: Configurations.
    :return:
    """

    if not configs['crop_image']:
        return

    if configs['data_format'] == "":
        if configs['output_dir'] is not None and len(configs['output_dir']) != 0:
            os.makedirs(configs['output_dir'], exist_ok=True)
        else:
            configs['output_dir'] = configs['image_dir']
    if configs['data_format'] == "hdf5":
        if configs['output_hdf5'] is not None and len(configs['output_hdf5']) != 0:
            with h5py.File(configs['output_hdf5'], 'a') as f:
                pass
                # f.close()
        else:
            configs['output_hdf5'] = configs['image_hdf5']

    # Detect uncropped images
    path_to_uncropped_images = detect_uncropped_images(configs)

    # Crop original images without a cropped version and save in dataset file.
    crop_image(configs, path_to_uncropped_images)
