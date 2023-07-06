import argparse
import os
from utils import make_directory, extract_zip, extract_tar
import torch
import sys


class BioScan_Configurations():

    def __init__(self, exp_ID):
        """
        This class handles basic configurations of the BIOSCAN-1M Insect Dataset and experiments.
        """

        self.dataset_names = {'1': 'BIOSCAN_Insect_Dataset',
                              '2': 'BIOSCAN_Insect_Diptera_Dataset',
                              }

        self.experiment_names = ['large_diptera_family',
                                 'medium_diptera_family',
                                 'small_diptera_family',
                                 'large_insect_order',
                                 'medium_insect_order',
                                 'small_insect_order',
                                 ]

        self.exp = self.experiment_names[exp_ID]
        self.group_level = get_group_level(exp_name=self.exp)
        self.max_num_sample = 0
        if self.exp in self.experiment_names[2:4]:
            self.max_num_sample = 200000
        elif self.exp in self.experiment_names[4:6]:
            self.max_num_sample = 50000

        self.data_formats = ["hdf5",    # hdf5_path
                             "folder",  # image_path (handles directory, tar, and zip)
                             ]


def get_group_level(exp_name=''):

    name = ''.join(list(exp_name)[-6:])
    if name == "family":
        group_level = 'family'
    elif name == "_order":
        group_level = 'order'
    else:
        print(f"experiment name is not verified: {exp_name}")
        return

    return group_level


def extract_package(image_path, data_format="folder"):

    if data_format != "folder" or os.path.isdir(image_path):
        return image_path

    data_format = os.path.splitext(os.path.basename(image_path))[1]
    path_to_extract = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0])
    if data_format == "zip":
        extract_zip(zip_file=image_path, path=path_to_extract)
    elif data_format == "tar":
        extract_tar(tar_file=image_path, path=path_to_extract)
    else:
        sys.exit("Wrong data_format: " + data_format + " does not exist.")

    path_to_images = os.path.join(path_to_extract,
                                  f"bioscan/images/{os.path.splitext(os.path.basename(image_path))[0]}")
    return path_to_images


def set_configurations(configs=None):

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    print("\nSetting Hyper-parameters of the Experiment ...")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--date_time', type=str, default=configs["date_time"], help='Data & time of the experiment',
                        required=False)

    # #### Base Settings ######
    parser.add_argument('--exp_name', type=str, default=configs["exp_name"], help='Name of the experiment',
                        required=False)
    parser.add_argument('--dataset_name', type=str, default=configs["dataset_name"], help='Name of the dataset.',
                        required=False)
    parser.add_argument('--group_level', type=str, default="order", help='Taxonomic group ranking.')
    parser.add_argument('--data_format', type=str, default=configs["data_format"], help='Format of the dataset.',
                        required=False)
    parser.add_argument('--best_model', type=str, help='directory where best results saved (inference/test mode).',
                        default=configs["best_model"], required=False)
    parser.add_argument('--chunk_length', type=int, default=10000, help='Chunk length: number of images of each patch.',
                        required=False)
    parser.add_argument('--chunk_num', type=int, default=0, help='set the data chunk number.', required=False)

    # #### Path Settings ######
    parser.add_argument('--log', type=str, default="runs", help='Path to the log file.', required=False)
    parser.add_argument('--download_path', type=str, help='Path to the download directory.',
                        default='', required=False)
    parser.add_argument('--dataset_path', type=str, help='Path to the BIOSCAN Dataset root.',
                        default=configs["dataset_path"], required=False)
    parser.add_argument('--metadata_path', type=str, default=configs["metadata_path"],
                        help="Path to the metadata of the dataset.", required=False)
    parser.add_argument('--results_path', type=str, help='Path to save results.', default=configs["results_path"],
                        required=False)
    parser.add_argument('--image_path', type=str, help='Path to the individual RGB images (if any).',
                        default=configs["image_path"], required=False)
    parser.add_argument('--hdf5_path', type=str, help='Path to the HDF5 files.', default=configs["hdf5_path"],
                        required=False)
    # The following needs to be set
    parser.add_argument('--resized_image_path', type=str, default=None, help="Path to the resized images.")
    parser.add_argument('--resized_hdf5_path', type=str, default=None, help="Path to the resized HDF5.")
    parser.add_argument('--cropped_image_path', type=str, default=None, help="Path to the cropped images.")
    parser.add_argument('--cropped_hdf5_path', type=str, help='Path to the HDF5 files of the CROPPED images.',
                        default=None)
    parser.add_argument('--resized_cropped_image_path', type=str, default=None,
                        help="Path to the cropped resized images.")
    parser.add_argument('--resized_cropped_hdf5_path', type=str, help='Path to the HDF5 files of the CROPPED images.',
                        default=None)

    # #### Condition Settings #####
    parser.add_argument('--make_split', help='Whether to split dataset into train, validation and test sets?',
                        default=False, action='store_true')
    parser.add_argument('--print_statistics', help='Whether to print dataset statistics?',
                        default=False, action='store_true')
    parser.add_argument('--print_split_statistics', help='Whether to print dataset split statistics?',
                        default=False, action='store_true')
    parser.add_argument('--loader', help='Whether to create dataloader?',
                        default=False, action='store_true')
    parser.add_argument('--train', help='Whether to train the model?',
                        default=False, action='store_true')
    parser.add_argument('--test', help='Whether to test the model?',
                        default=False, action='store_true')
    parser.add_argument('--crop_image', help='Whether to crop dataset images?',
                        default=False, action='store_true')
    parser.add_argument('--no_transform', default=False, action='store_true',
                        help='Not using transformation in dataloader?')
    parser.add_argument('--cropped', default=True, action='store_true',
                        help='Using cropped images?')

    # ####### Data Download #####
    parser.add_argument('--ID_mapping_path', type=str, default="dataset/bioscan_1M_insect_dataset_file_ID_mapping.txt",
                        help="Path to the directory where file ID mapping is saved.")
    parser.add_argument('--download_path', type=str, default=configs["download_path"],
                        help="Path to the dataset files downloaded from drive.")
    parser.add_argument('--file_to_download', type=str, default=None,
                        help="File to download from drive.")

    # ####### Data Split and Subset Creation #####
    parser.add_argument('--max_num_sample', type=int, default=50000,
                        help='Number of samples of each subset.',
                        required=False)
    parser.add_argument('--experiment_names', type=str, default=configs["experiment_names"],
                        help="Name of experiments conducted in BIOSCAN Paper.", required=False)
    # #### Preprocessing: Cropping Settings ######
    parser.add_argument('--checkpoint_path', type=str, default=configs["checkpoint_path"],
                        help="Path to the checkpoint.", required=False)
    parser.add_argument('--use_metadata', type=str, default=False,
                        help='If using metadata for cropping?')
    parser.add_argument('--crop_ratio', type=float, default=1.4, help="Scale the bbox to crop larger or small area.")
    parser.add_argument('--equal_extend', default=True,
                        help="Extend cropped images in the height and width with the same length.", action="store_true")

    # #### Training Settings ######
    parser.add_argument('--seed', type=int, default=1, help='Set the seed for reproducibility', required=False)
    parser.add_argument('--n_epochs', type=int, default=100, required=False)
    parser.add_argument('--epoch_decay', nargs='+', type=int, default=[20, 25], required=False)
    parser.add_argument('--mu', type=float, default=0.0001, help='weight decay parameter', required=False)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum', required=False)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate to use', required=False)
    parser.add_argument('--batch_size', type=int, default=32, help='default is 32', required=False)
    parser.add_argument('--image_size', type=int, default=256, required=False)
    parser.add_argument('--crop_size', type=int, default=224, required=False)
    parser.add_argument('--num_workers', type=int, default=4, required=False)
    parser.add_argument('--k', nargs='+', help='value of k for computing the top-k loss and computing top-k accuracy',
                        default=[1, 3, 5, 10], type=int, required=False)
    parser.add_argument('--pretrained', default=True, action='store_true', required=False)
    parser.add_argument('--vit_pretrained', type=str, default=configs["vit_pretrained"],
                        help="Path to the checkpoint.", required=False)
    parser.add_argument('--loss', type=str, help='decide which loss to use during training', default='CE',
                        choices=["CE", "Focal"])

    # #### Model Settings #####
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                            'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                            'mobilenet_v2', 'inception_v3', 'alexnet', 'squeezenet',
                                            'shufflenet', 'wide_resnet50_2', 'wide_resnet101_2',
                                            'vgg11', 'mobilenet_v3_large', 'mobilenet_v3_small',
                                            'inception_resnet_v2', 'inception_v4', 'efficientnet_b0',
                                            'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                            'efficientnet_b4', 'vit_base_patch16_224', 'vit_small_patch16_224'],
                        default='resnet50', help='choose the model you want to train on')

    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available(), )
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def save_configs(datetime, configs, log_dir=None):

    info = f"Configurations of the Experiments Run on {datetime}\n"
    for item in configs.keys():
        info += f'{item}:{configs[item]}\n'

    with open(log_dir + f"{configs['exp_name']}_configs.txt", "w") as fp:
        fp.write(info)
    fp.close()


def make_path_configs(configs):

    if configs["train"]:
        save_dir = os.path.join(os.getcwd(), configs["results_path"])
        save_dir += "/{timestamp:s}_{dataset:s}_{loss:s}_{model:s}/".format(timestamp=configs["date_time"],
                                                                            dataset=configs['exp_name'],
                                                                            loss=configs['loss'],
                                                                            model=configs['model'])
        make_directory(save_dir)
        save_configs(configs["date_time"], configs, log_dir=save_dir)
        configs["results_path"] = save_dir

    return configs



