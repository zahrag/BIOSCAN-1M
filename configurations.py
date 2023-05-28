import argparse
import os
from utils import make_directory
import torch


class BioScan_Configurations():
    def __init__(self):
        """
        This class handles basic configurations of the BioScan Dataset and experiments.
        """

        self.taxonomic_group_levels = {'0': 'domain', '1': 'kingdom', '2': 'phylum', '3': 'class', '4': 'order',
                                       '5': 'family', '6': 'subfamily', '7': 'tribe', '8': 'genus', '9': 'species',
                                       '10': 'subspecies', '11': 'name'}

        self.dataset_names = {'1': '200K_dataset',
                              '2': '200K_insect_dataset',
                              '3': '200K_insect_diptera_dataset',

                              '4': 'bioscan_dataset',
                              '5': 'bioscan_insect_dataset',
                              '6': 'bioscan_insect_diptera_dataset',

                              '7': '80K_dataset',
                              '8': '80K_insect_dataset',
                              '9': '80K_insect_diptera_dataset', }

        self.data_formats = ["folder", "hdf5", "tar"]


def set_configurations(config=None):

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    print("\n Setting Hyper-parameters of the Experiment ...")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--date_time', type=str, default=config["date_time"], help='Data & time of the experiment',
                        required=False)

    # #### Path Settings ######
    parser.add_argument('--exp_name', type=str, default=config["exp_name"], help='Name of the experiment',
                        required=False)
    parser.add_argument('--dataset_name', type=str, default=config["dataset_name"], help='Name of the dataset.',
                        required=False)
    parser.add_argument('--group_level', type=str, default=config["group_level"], help='Taxonomic group ranking.',
                        required=False)
    parser.add_argument('--data_format', type=str, default=config["data_format"], help='Format of the dataset.',
                        required=False)
    parser.add_argument('--log', type=str, default="runs", help='Path to the log file.', required=False)
    parser.add_argument('--download_path', type=str, help='Path to the download directory.',
                        default='', required=False)
    parser.add_argument('--dataset_path', type=str, help='Path to the BioScan Dataset.',
                        default=config["dataset_path"], required=False)
    parser.add_argument('--metadata_path', type=str, default=config["metadata_path"],
                        help="Path to the metadata of the dataset.", required=False)
    parser.add_argument('--metadata_path_train', type=str, default=config["metadata_path_train"],
                        help="Path to the metadata of the Train set.", required=False)
    parser.add_argument('--metadata_path_validation', type=str, default=config["metadata_path_val"],
                        help="Path to the metadata of the Validation set.", required=False)
    parser.add_argument('--metadata_path_test', type=str, default=config["metadata_path_test"],
                        help="Path to the metadata of the Test set.", required=False)

    parser.add_argument('--image_path', type=str, help='Path to the individual RGB images (if any).',
                        default=config["image_path"], required=False)
    parser.add_argument('--hdf5_path', type=str, help='Path to the HDF5 files.', default=config["hdf5_path"],
                        required=False)
    parser.add_argument('--results_path', type=str, help='Path to save results.', default=config["results_path"],
                        required=False)
    parser.add_argument('--no_transform', default=False, action='store_true',
                        help='Not using transformation in dataloader?', required=False)
    parser.add_argument('--cropped', default=True, action='store_true',
                        help='Using cropped images?',
                        required=False)

    # #### Condition Settings #####
    parser.add_argument('--download', help='Whether to download from drive?',
                        default=False, action='store_true')
    parser.add_argument('--make_split', help='Whether to split dataset into train, validation and test sets?',
                        default=config["make_split"], action='store_true')
    parser.add_argument('--print_statistics', help='Whether to print dataset statistics?',
                        default=config["print_statistics"], action='store_true')
    parser.add_argument('--print_split_statistics', help='Whether to print dataset split statistics?',
                        default=config["print_split_statistics"], action='store_true')
    parser.add_argument('--loader', help='Whether to create dataloader?',
                        default=config["dataloader"], action='store_true')
    parser.add_argument('--train', help='Whether to train the model?',
                        default=config["train"], action='store_true')
    parser.add_argument('--test', help='Whether to test the model?',
                        default=config["test"], action='store_true')
    parser.add_argument('--crop_image', help='Whether to crop dataset images?',
                        default=config["crop_image"], action='store_true')

    # #### Preprocessing: Cropping Settings ######
    parser.add_argument('--read_format', type=str, default="hdf5",
                        help='Format of the dataset, we want to read from.', required=False)
    parser.add_argument('--write_format', type=str, default="hdf5",
                        help='Format of the dataset, we write cropped images into.', required=False)
    parser.add_argument('--chunk_length', type=int, default=10000, help='Chunk length: number of images of each patch.',
                        required=False)
    parser.add_argument('--chunk_num', type=int, default=0, help='set the data chunk number.', required=False)
    parser.add_argument('--meta_path', type=str, default="", help="path to the meta directory")
    parser.add_argument('--image_hdf5', type=str, default="", help="path to the image hdf5y")
    parser.add_argument('--output_dir', type=str, default="", help="path to the image directory")
    parser.add_argument('--output_hdf5', type=str, default="", help="path to the image hdf5y")
    parser.add_argument('--checkpoint_path', type=str, default="", help="Path to the checkpoint.")
    parser.add_argument('--crop_ratio', type=float, default=1.4, help="Scale the bbox to crop larger or small area.")
    parser.add_argument('--equal_extend', default=True, help="Extend cropped images in the height and width with the same length.", action="store_true")

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
    parser.add_argument('--loss', type=str, help='decide which loss to use during training', default=config["loss"],
                        required=False, choices=["CE", "Focal"])

    # #### Model Settings #####
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                            'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                            'mobilenet_v2', 'inception_v3', 'alexnet', 'squeezenet',
                                            'shufflenet', 'wide_resnet50_2', 'wide_resnet101_2',
                                            'vgg11', 'mobilenet_v3_large', 'mobilenet_v3_small',
                                            'inception_resnet_v2', 'inception_v4', 'efficientnet_b0',
                                            'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                            'efficientnet_b4', 'vit_base_patch16_224', 'vit_small_patch16_224'],
                        default=config["model"], help='choose the model you want to train on', required=False)

    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available(), )
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def save_configs(datetime, config, log_dir=None):

    info = f"Configurations of the Experiments Run on {datetime}\n"
    for item in config.keys():
        info += f'{item}:{config[item]}\n'

    with open(log_dir + f"{config['exp_name']}_configs.txt", "w") as fp:
        fp.write(info)
    fp.close()


def make_path_configs(config):

    if config["train"]:
        save_dir = os.path.join(os.getcwd(), config["results_path"])
        save_dir += "/{timestamp:s}_{dataset:s}/".format(timestamp=config["date_time"],
                                                         dataset=config['dataset_name'])
        make_directory(save_dir)
        save_configs(config["date_time"], config, log_dir=save_dir)
        config["results_path"] = save_dir
        config["exp_name"] = config["dataset_name"]

    return config


