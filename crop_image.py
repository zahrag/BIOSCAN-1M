import os
import pickle
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import make_directory

from transformers import DetrFeatureExtractor
from crop_tool.util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox
from crop_tool.scripts.crop_images import expand_image
from crop_tool.model.detr import load_model_from_ckpt

from utils import make_hdf5

import matplotlib.pyplot as plt


class CustomArg:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def save_cropped_image(args, original_image_name, cropped_image, cropped_imgs_hdf5, cropped_imgs_path):
    """
    This function saves the cropped image in the corresponding file format of the dataset.
    :param args:
    :param original_image_name:
    :param cropped_image:
    :param cropped_imgs_hdf5:
    :param cropped_imgs_path:
    :return:
    """

    # Save individual CROPPED images in a directory different from Original images.
    if args['write_format'] == "folder":
        cropped_image.save(os.path.join(cropped_imgs_path, "CROPPED_" + original_image_name))

    # Save cropped images in HDF5 file format named: CROPPED_HDF5_(dataset_name)
    if args['write_format'] == "hdf5":
        cropped_imgs_hdf5.create_dataset("CROPPED_" + original_image_name, data=cropped_image)


def crop_image(args, original_images_path, model, feature_extractor, orig_imgs_hdf5, cropped_imgs_hdf5, cropped_imgs_path):

    """
    :param args:
    :param original_images: Path if images are in different directories, Name if all images are in the same directory.
    :param model:
    :param feature_extractor:
    :param orig_imgs_hdf5:
    :param orig_imgs_path:
    :param cropped_imgs_hdf5:
    :param cropped_imgs_path:
    :return:
    """

    # Open an image file: original image to be cropped.
    for image_path in tqdm(original_images_path):
        image_name = os.path.basename(image_path)
        if args['read_format'] == "folder":
            if os.path.isfile(image_path):
                try:
                    image = Image.open(image_path)
                except:
                    print("Image not found in: " + image_path)
                    exit(1)

        # get original images saved in a HDF5 file format
        elif args['read_format'] == "hdf5":
            data = np.asarray(orig_imgs_hdf5[image_name])
            image = Image.fromarray(data)

        encoding = feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0).cuda()
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
        show_cropped_imgs = True
        if show_cropped_imgs:
            plt.figure()
            plt.imshow(cropped_img)
            plt.show()

        # Save the cropped image
        save_cropped_image(args, image_name, cropped_img, cropped_imgs_hdf5, cropped_imgs_path)


def detect_uncropped_images(args, original_img_path, cropped_imgs_path, cropped_imgs_hdf5):

    """
    This function detects the images in the dataset file, which do not have a cropped version.
    :param args:
    :param original_img_path: Path to where original images are saved.
    :param cropped_imgs_path: Path to where original images are/will be saved.
    :param cropped_imgs_hdf5: HDF5 file of the cropped images.
    :return:
    """

    # Get original images that you want to crop!
    df = pd.read_csv(f"{args['mata_path']}/{args['dataset_name']}_train_metadata.tsv", sep='\t', low_memory=False)
    image_names = df['image_file'].to_list()

    # Check if the images are already cropped!
    if args['read_format'] == "folder":
        uncropped_imgs_path = [f'{original_img_path}/{img}' for img in image_names if f'CROPPED_{img}' not in os.listdir(cropped_imgs_path)]

    elif args['read_format'] == "hdf5":
        uncropped_imgs_path = [img for img in image_names if f'CROPPED_{img}' not in cropped_imgs_hdf5.keys()]

    else:
        raise RuntimeError("Please select a format to detect uncropped images: folder or hdf5!")

    return uncropped_imgs_path


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

    # ############################# Reading original Images ###################################
    orig_img_hdf5 = None
    orig_img_path = ''
    if args['read_format'] == "hdf5":
        hdf5_path = f"{args['hdf5_dir']}/HDF5_{args['dataset_name']}"
        hdf5_file = h5py.File(hdf5_path, 'r')
        orig_img_hdf5 = hdf5_file[args['dataset_name']]

    elif args['read_format'] == "folder":
        orig_img_path = f"{args['image_dir']}"

    else:
        raise RuntimeError("Please select a format to read ORIGINAL images: folder or hdf5!")

    # ############################# Writing cropped images ###################################
    cropped_img_hdf5 = None
    cropped_img_path = ''
    if args['write_format'] == "hdf5":
        hdf5_cropped_path = f"{args['dataset_dir']}/HDF5_{args['dataset_name']}_CROPPED"
        if os.path.isfile(hdf5_cropped_path):
            print(f"{hdf5_cropped_path} already exists!")
            # Open hdf5 file in an append mode
            hdf5_file = h5py.File(hdf5_cropped_path, 'a')
            cropped_img_hdf5 = hdf5_file[args['dataset_name']]
        else:
            # Create the file in the write mode
            hdf5_file = make_hdf5(args['date_time'],
                                  dataset_name=args['dataset_name'], path=hdf5_cropped_path,
                                  data_typ="Cropped")
            cropped_img_hdf5 = hdf5_file[args['dataset_name']]
            print(f"{hdf5_cropped_path} is created!")

    elif args['write_format'] == "folder":
        cropped_img_path = f"{args['dataset_dir']}/{args['dataset_name']}/{args['dataset_name']}_CROPPED_images"
        make_directory(cropped_img_path)

    else:
        raise RuntimeError("Please select a format to write CROPPED images: folder or hdf5!")

    # ############################# Get Uncropped Images ###################################
    uncropped_images = detect_uncropped_images(args, orig_img_path, cropped_img_path, cropped_img_hdf5)

    # Set essentials of crop tool
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = load_model_from_ckpt(args)
    model.cuda()

    # Crop original images without a cropped version and save in dataset file.
    crop_image(args, uncropped_images, model, feature_extractor, orig_img_hdf5, cropped_img_hdf5, cropped_img_path)
