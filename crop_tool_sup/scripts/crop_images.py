import argparse
import os
import sys
from os.path import dirname, abspath
import numpy as np
from tqdm import tqdm
from transformers import DetrFeatureExtractor
from PIL import Image, ImageDraw, ImageOps

project_dir = dirname(dirname(abspath(__file__)))
sys.path.append(project_dir)
from model.detr import load_model_from_ckpt
from util.visualize_and_process_bbox import get_bbox_from_output, scale_bbox


def expand_image(args, image, size, direction):
    border_color = (args.background_color_R, args.background_color_G, args.background_color_B)
    border_image = None
    if direction == 'left':
        border_image = ImageOps.expand(image, border=(size, 0, 0, 0), fill=border_color)
    elif direction == 'top':
        border_image = ImageOps.expand(image, border=(0, size,0 , 0), fill=border_color)
    elif direction == 'right':
        border_image = ImageOps.expand(image, border=(0, 0, size, 0), fill=border_color)
    elif direction == 'bottom':
        border_image = ImageOps.expand(image, border=(0, 0, 0, size), fill=border_color)
    else:
        exit("Wrong expand direction.")

    return border_image



def rotate_image_and_bbox_if_necesscary(image, left, top, right, bottom):
    image_size = image.size
    image = image.rotate(90, expand=True)
    new_left = top
    new_right = bottom
    new_bottom = image_size[0] - left
    new_top = image_size[0] - right

    return image, new_left, new_top, new_right, new_bottom


def change_size_to_4_3(left, top, right, bottom):
    width = right - left
    height = bottom - top
    if width < height / 3 * 4:
        extend_length = height / 3 * 4 - width
        left = left - extend_length / 2
        right = right + extend_length - extend_length / 2

    elif height < width / 4 * 3:
        extend_length = width / 4 * 3 - height
        top = top - extend_length / 2
        bottom = bottom + extend_length - extend_length / 2
    return left, top, right, bottom


def crop_image(args, model, feature_extractor):
    """
    Crop and save images based on the predicted bounding boxes from the model.
    :param model: Detr model that loaded from the checkpoint.
    :param feature_extractor: A ResNet50 model as a standard image extractor.
    """
    for filename in tqdm(os.listdir(args.input_dir)):
        f = os.path.join(args.input_dir, filename)
        if os.path.isfile(f):
            try:
                image = Image.open(f)
            except:
                print("Image not found in: " + f)
                exit(1)
            encoding = feature_extractor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze().unsqueeze(0)
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
            bbox = get_bbox_from_output(outputs, image).detach().numpy()
            bbox = np.round(bbox, 0)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            if args.show_bbox:
                draw = ImageDraw.Draw(image)
                draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=args.width_of_bbox)
            left, top, right, bottom = scale_bbox(args, left, top, right, bottom)

            image_size = image.size

            if args.fix_ratio:
                width = right - left
                height = bottom - top

                if height > width and args.rotate_image:
                    image, left, top, right, bottom = rotate_image_and_bbox_if_necesscary(image, left, top, right,
                                                                                          bottom)
                    image_size = image.size

                left, top, right, bottom = change_size_to_4_3(left, top, right, bottom)
                left = round(left)
                top = round(top)
                right = round(right)
                bottom = round(bottom)
                # if left < 0 or top < 0 or right > image_size[0] or bottom > image_size[1]:
                #     print("Please crop this image manually: " + f)

            # cropped_img = image.crop((max(left, 0), max(top, 0), min(right, image_size[0]), min(bottom, image_size[1])))

            # Check if width is smaller than the bbox

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


            cropped_img.save(os.path.join(args.output_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Folder that contains the original images.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the checkpoint.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Number of images in each batch.")
    parser.add_argument('--output_dir', type=str, default="cropped_image",
                        help="Folder that will contain the cropped images.")
    parser.add_argument('--crop_ratio', type=float, default=1.4,
                        help="Scale the bbox to crop larger or small area.")
    parser.add_argument('--show_bbox', default=False,
                        action='store_true')
    parser.add_argument('--width_of_bbox', type=int, default=3,
                        help="Define the width of the bound of bounding boxes.")
    parser.add_argument('--fix_ratio', default=False,
                        action='store_true', help='Further extent the image to make the ratio in 4:3.')
    parser.add_argument('--equal_extend', default=False,
                        action='store_true', help='Extand equal size in both height and width.')
    parser.add_argument('--rotate_image', default=False,
                        action='store_true', help='Rotate the insect to fit 4:3 naturally.')
    parser.add_argument('--background_color_R', type=int, default=234,
                        help="Define the background color's R value.")
    parser.add_argument('--background_color_G', type=int, default=242,
                        help="Define the background color's G value.")
    parser.add_argument('--background_color_B', type=int, default=245,
                        help="Define the background color's B value.")



    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    model = load_model_from_ckpt(args)

    crop_image(args, model, feature_extractor)
