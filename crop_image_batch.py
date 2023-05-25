import os
import io
import pickle
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import make_directory
import matplotlib.pyplot as plt

from transformers import DetrFeatureExtractor
from crop_tool_sup.util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox
from crop_tool_sup.scripts.crop_images import expand_image
from crop_tool_sup.model.detr import load_model_from_ckpt


class CustomArg:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def open_pickle(path):

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    results = objects[0]

    return results


def make_hdf5(date_time, path='', data_typ=''):

    with h5py.File(path, 'w') as hdf5:

        dataset = hdf5.create_group('bioscan_dataset')
        dataset.attrs['Description'] = f'BioScan Dataset: {data_typ} Images'
        dataset.attrs['Copyright Holder'] = 'CBG Photography Group'
        dataset.attrs['Copyright Institution'] = 'Centre for Biodiversity Genomics (email:CBGImaging@gmail.com)'
        dataset.attrs['Photographer'] = 'CBG Robotic Imager'
        dataset.attrs['Author'] = 'Zahra Gharaee'
        dataset.attrs['Date'] = date_time


def save_cropped_image(args, original_image_name, cropped_image, cropped_hdf5_path, cropped_imgs_path):
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
        cropped_image.save(os.path.join(cropped_imgs_path, f"CROPPED_{original_image_name}"))

    # Save cropped images in HDF5 file format
    if args['write_format'] == "hdf5":
        with h5py.File(cropped_hdf5_path, 'a') as hdf5:
            dataset = hdf5['bioscan_dataset']
            binary_data_io = io.BytesIO()
            cropped_image.save(binary_data_io, format='JPEG')
            # Read the binary data from the BytesIO object
            binary_data = binary_data_io.getvalue()
            binary_data_np = np.frombuffer(binary_data, dtype=np.uint8)
            dataset.create_dataset(f"CROPPED_{original_image_name}", data=binary_data_np)


def crop_image(args, group_name, original_images_path, orig_imgs_hdf5, cropped_imgs_hdf5, cropped_imgs_path):

    """
    :param args:
    :param group_name:
    :param original_images_path:
    :param model:
    :param feature_extractor:
    :param orig_imgs_hdf5:
    :param cropped_imgs_hdf5:
    :param cropped_imgs_path:
    :return:
    """
    # If using not a dictionary args
    real_args = CustomArg(args)

    # Get essentials of the crop tool
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = load_model_from_ckpt(real_args)

    # Open an image file: original image to be cropped.
    cnt = 0
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
            with h5py.File(orig_imgs_hdf5, 'r') as hdf5:
                dataset = hdf5[group_name]
                data = np.array(dataset[image_name])
                image = Image.open(io.BytesIO(data))

        else:
            raise RuntimeError("Please select a format to write CROPPED images: folder or hdf5!")

        encoding = feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0)
        outputs = model(pixel_values=pixel_values, pixel_mask=None)
        bbox = get_bbox_from_output(outputs, image).detach().numpy()
        bbox = np.round(bbox, 0)
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        left, top, right, bottom = scale_bbox(real_args, left, top, right, bottom)
        real_args.background_color_R = 240
        real_args.background_color_G = 240
        real_args.background_color_B = 240
        if left < 0:
            border_size = 0 - left
            right = right - left
            left = 0
            image = expand_image(real_args, image, border_size, 'left')
        if top < 0:
            border_size = 0 - top
            bottom = bottom - top
            top = 0
            image = expand_image(real_args, image, border_size, 'top')
        if right > image.size[0]:
            border_size = right - image.size[0] + 1
            image = expand_image(real_args, image, border_size, 'right')
        if bottom > image.size[1]:
            border_size = bottom - image.size[1] + 1
            image = expand_image(real_args, image, border_size, 'bottom')
        cropped_img = image.crop((left, top, right, bottom))

        show_cropped_imgs = False
        if show_cropped_imgs:
            plt.figure()
            plt.imshow(cropped_img)
            plt.show()

        # Save the cropped image
        save_cropped_image(args, image_name, cropped_img, cropped_imgs_hdf5, cropped_imgs_path)
        cnt += 1


def detect_uncropped_images(args, original_img_path, cropped_imgs_path, cropped_imgs_hdf5):

    """
    This function detects the images in the dataset file, which do not have a cropped version.
    :param args:
    :param original_img_path:
    :param cropped_imgs_path:
    :param cropped_imgs_hdf5:
    :return:
    """

    # Get original images that you want to crop!
    metadata_path = ""
    df = pd.read_csv(metadata_path, sep='\t', low_memory=False)
    image_names = df['image_file'].to_list()

    # Check if the images are already cropped!
    if args['read_format'] == "folder":
        uncropped_imgs_path = [f'{original_img_path}/{img}' for img in image_names if f'CROPPED_{img}' not in os.listdir(cropped_imgs_path)]

    elif args['read_format'] == "hdf5":
        with h5py.File(cropped_imgs_hdf5, 'r') as hdf5:
            uncropped_imgs_path = [img for img in image_names if f'CROPPED_{img}' not in hdf5.keys()]
    else:
        raise RuntimeError("Please select a format to detect uncropped images: folder or hdf5!")

    return uncropped_imgs_path


def get_image_path_metadata(chunk_num, start_index=0, meta_file='', image_num=10000):

    print(f"\n\nChunk Number: {chunk_num}")

    df = pd.read_csv(meta_file, sep='\t', low_memory=False)
    all_images = df['image_file'].to_list()
    print(f"\nAll images: {len(all_images)}")
    partition_numbers = df['chunk_number'].to_list()

    if ((chunk_num+1) * image_num) > len(all_images):
        selected_images = all_images[chunk_num * image_num:]
        chunk_selected = partition_numbers[chunk_num * image_num:]
        print('This chunk is the last one!')
    else:
        selected_images = all_images[chunk_num * image_num:(chunk_num + 1) * image_num]
        chunk_selected = partition_numbers[chunk_num * image_num:(chunk_num + 1) * image_num]

    if all(item == chunk_selected[0] for item in chunk_selected) and chunk_selected[0] == start_index + chunk_num + 1:
        print(f"Number of selected images:{len(selected_images)}")
        return selected_images
    else:
        print(f"\nChunk number:{chunk_num} is NOT verified!")
        return


def run_crop_tool(args, group_name=''):
    """
    This function runs as offline preprocessing to detect the images without a cropped version in the dataset, and to
    create a cropped version and save it in the corresponding data format.
    :param args:
    :return:
    """

    if not args['crop_image']:
        return

    start_index = 0
    if args['chunk_num'] not in range(120):
        raise RuntimeError(f" {args['chunk_num']} NOT in range 0:119")

    # ############################# Reading Original Images ###################################
    foldr_orig_path = ''
    hdf5_orig_path = ''
    if args['read_format'] == "hdf5":
        hdf5_orig_path = f"{args['hdf5_path']}/BioScan_HDF5_Part{start_index + args['chunk_num'] + 1}"
        if os.path.isfile(hdf5_orig_path):
            print(f'\nPath to Original HDF5 file:\n{hdf5_orig_path}')
        else:
            RuntimeError(f"Path to Original HDF5 file:\n{hdf5_orig_path} does NOT exist!")

    elif args['read_format'] == "folder":
        foldr_orig_path = ''

    else:
        raise RuntimeError("Please select a format to read ORIGINAL images: folder or hdf5!")

    # ############################# Writing Cropped Images ###################################
    hdf5_cropped_path = ''
    folder_cropped_path = ''
    if args['write_format'] == "hdf5":
        hdf5_cropped_path = f"{args['cropped_hdf5_path']}/BioScan_HDF5_CROPPED_Part{start_index + args['chunk_num'] + 1}"
        print(f'\nPath to Cropped HDF5 file:\n{hdf5_cropped_path}')
        if os.path.isfile(hdf5_cropped_path):
            raise RuntimeError(f"Path to Cropped HDF5 file {hdf5_cropped_path} already exists!")
        else:
            make_hdf5(args['date_time'], path=hdf5_cropped_path, data_typ="Cropped")

    elif args['write_format'] == "folder":
        folder_cropped_path = ""
        make_directory(folder_cropped_path)

    else:
        raise RuntimeError("Please select a format to write CROPPED images: folder or hdf5!")

    # ############################# Get Uncropped Images ###################################
    # ####### Detect Uncropped Images
    # uncropped_images = detect_uncropped_images(args, foldr_orig_path, folder_cropped_path, hdf5_cropped_path)

    # ####### Get Uncropped Images from Metadata
    uncropped_images = get_image_path_metadata(args['chunk_num'], start_index=start_index,
                                               meta_file=args['metadata_path'], image_num=args['chunk_length'])
    print(f"Number of Uncropped Images:{len(uncropped_images)}")

    # Crop original images without a cropped version and save in dataset file.
    crop_image(args, group_name, uncropped_images, hdf5_orig_path, hdf5_cropped_path, folder_cropped_path)
    print(f"Cropping Completed for the Chunk:{args['chunk_num']} and the file is save in:\n{hdf5_cropped_path}")
