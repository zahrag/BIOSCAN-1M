import os
import sys
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

from transformers import DetrFeatureExtractor
from crop_tool.util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox
from crop_tool.scripts.crop_images import expand_image
from crop_tool.model.detr import load_model_from_ckpt


class CustomArg:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def save_cropped_image(args, img, cropped_img, output_hdf5, file):
    """
    This function saves the cropped image in the corresponding file format of the dataset.
    :param args:
    :param img: Original image file.
    :param cropped_img: cropped image.
    :param output_hdf5: HDF5 output file if already exits.
    :param file: HDF5 file if just created.
    :return:
    """

    if args['data_format'] == "":
        if args['output_dir'] is not None:
            cropped_img.save(os.path.join(args['output_dir'], "CROPPED_" + os.path.basename(img)))
        else:
            cropped_img.save(os.path.join(os.path.dirname(img), "CROPPED_" + os.path.basename(img)))

    if args['data_format'] == "hdf5":
        if args['output_dir'] is not None:
            output_hdf5.create_dataset("CROPPED_" + img, data=cropped_img)
        else:
            file[args['dataset_name']].create_dataset("CROPPED_" + img, data=cropped_img)


def crop_image(args_dict, original_images):
    """
    This function does the cropping.

    :param args_dict:
    :param original_images: path to the list of uncropped images.
    :return:
    """
    if isinstance(args_dict, dict):
        args = CustomArg(d=args_dict)
    else:
        args = args_dict
    if args['checkpoint_path'] is None:
        args['checkpoint_path'] = args_dict['cropping_checkpoint']
    if argsimage_hdf5 is not None:
        args['hdf5_dir'] = args['image_hdf5']
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = load_model_from_ckpt(args)
    output_hdf5 = None
    if args['data_format'] == "hdf5":
        file = h5py.File(args['hdf5_dir'], 'a')
        if args['output_hdf5'] is not None:
            output_hdf5 = h5py.File(args['output_hdf5'], 'w')
    for orig_img in tqdm(original_images):
        if args['data_format'] == "":
            if os.path.isfile(orig_img):
                try:
                    image = Image.open(orig_img)
                except:
                    print("Image not found in: " + orig_img)
                    exit(1)
        elif args['data_format'] == "hdf5":
            data = np.asarray(file[args['dataset_name']][orig_img])
            image = Image.fromarray(data)
        encoding = feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0)
        outputs = model(pixel_values=pixel_values, pixel_mask=None)
        bbox = get_bbox_from_output(outputs, image).detach().numpy()
        bbox = np.round(bbox, 0)
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        left, top, right, bottom = scale_bbox(args, left, top, right, bottom)
        args['background_color_R'] = 255
        args['background_color_G'] = 255
        args['background_color_B'] = 255
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
        save_cropped_image(args, orig_img, cropped_img, output_hdf5, file)


def detect_uncropped_images(args):

    """
    This function detects the images in the dataset file, which do not have a cropped version.
    :param args:
    :return: path to the uncropped images of the dataset
    """

    list_of_uncropped_image_path = []
    df = pd.read_csv(args['mata_path'], sep='\t', low_memory=False)
    image_names = df['image_file'].to_list()

    if args['data_format'] == "":
        for img in image_names:
            curr_image_dir = os.path.join(args['image_dir'], img)
            curr_cropped_image_dir = os.path.join(args['image_dir'], "CROPPED_" + img)
            if not os.path.isfile(curr_image_dir):
                sys.exit(curr_image_dir + "does not exit")
            if not os.path.isfile(curr_cropped_image_dir):
                list_of_uncropped_image_path.append(curr_image_dir)
    elif args['data_format'] == "hdf5":
        file = h5py.File(args['image_hdf5'], 'a')
        keys = file[args['dataset_name']].keys()
        for i in image_names:
            if i not in keys:
                sys.exit(i + "does not exit")
            if 'CROPPED_' + i not in keys:
                list_of_uncropped_image_path.append(i)
    else:
        sys.exit("Wrong data_format: " + args['data_format'] + " does not exist.")

    return list_of_uncropped_image_path


def run_crop_tool(args, crop_images=False):
    """
    This function runs as offline preprocessing to detect the images without a cropped version in the dataset, and to
    create a cropped version and save it in the corresponding data format.
    :param args:
    :param crop_images: If crop images?
    :return:
    """

    if not crop_images:
        return

    if args['data_format'] == "" and args['output_dir'] is not None:
        os.makedirs(args['data_format'], exist_ok=True)
    if args['data_format'] == "hdf5" and args['output_hdf5'] is not None:
        with h5py.File(args['output_hdf5'], 'a') as f:
            pass
            # f.close()

    # Detect uncropped images
    path_to_uncropped_images = detect_uncropped_images(args)

    # Crop original images without a cropped version and save in dataset file.
    crop_image(args, path_to_uncropped_images)
