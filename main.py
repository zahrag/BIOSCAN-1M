import datetime
import argparse
from BioScanDataSet import show_statistics
from bioscan_datadownload import download_data_files
from bioscan_datasplit import make_split
from bioscan_dataloader import get_dataloader
from train import train
from test import test
from utils import save_configs, make_directory
import torch
import os


def make_configurations():
    """
        This function delivers model configurations ...
        If you want to set an argument from terminal then set required=True for it.
        """

    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    config = {
        "download_dir": "",               # where to download data files
        "dataset_dir": "",                # root directory of the dataset, where images and dataframe files are saved
        "results_dir": "",                # where results are saved (set for evaluation of the trained model)
        "dataset_name": "small_dataset",  # Name of the dataset, exe., small_dataset, medium_dataset, big_dataset
        "data_type": "order",
        "data_format": "hdf5",            # the file format used (exe., hdf5)
        "exp_name": "bioscan_classification",
        "download": False,
        "make_split": False,
        "print_statistics": True,
        "dataloader": True,
        "train": True,
        "test": False,
        "batch_size": 32,
        "image_size": 256,
        "crop_size": 224,
        "num_workers": 4,
        "seed": 1,
        "n_epochs": 10,
        "epoch_decay": [20, 25],
        "momentum": 0.9,
        "mu": 0.0001,
        "lr": 0.01,
        "k": [1, 3, 5, 10],
        "model": "resnet50",
    }

    if config["test"]:
        return config

    save_dir = os.path.join(os.getcwd(), 'results')
    save_dir += "/{timestamp:s}/".format(timestamp=timestamp)
    make_directory(save_dir)
    save_configs(timestamp, config, log_dir=save_dir)
    args["results_dir"] = save_dir

    return config


if __name__ == '__main__':

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    # #### get configurations ######
    config = make_configurations()

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # #### Path Settings ######
    parser.add_argument('--exp_name', type=str, default=config["dataset_name"], help='name of the experiment',
                        required=False)
    parser.add_argument('--dataset_name', type=str, default=config["dataset_name"], help='name of the dataset',
                        required=False)
    parser.add_argument('--data_type', type=str, default=config["data_type"], help='type of the data', required=False)
    parser.add_argument('--data_format', type=str, default=config["data_format"], help='format of the dataset',
                        required=False)
    parser.add_argument('--log', type=str, default="runs", help='Path to the log file', required=False)
    parser.add_argument('--download_dir', type=str, help='Directory to download our dataset',
                        default=config["download_dir"], required=False)
    parser.add_argument('--dataset_dir', type=str, help='Directory of our dataset', default=config["dataset_dir"],
                        required=False)
    parser.add_argument('--results_dir', type=str, help='Directory to save results', default=config["results_dir"],
                        required=False)

    # #### Condition Settings #####
    parser.add_argument('--download', help='Whether to download from drive?',
                        default=config["download"], action='store_true')
    parser.add_argument('--make_split', help='Whether to split dataset into train, validation and test sets?',
                        default=config["make_split"], action='store_true')
    parser.add_argument('--print_statistics', help='Whether to print dataset statistics?',
                        default=config["print_statistics"], action='store_true')
    parser.add_argument('--loader', help='Whether to generate dataloader?',
                        default=config["dataloader"], action='store_true')
    parser.add_argument('--train', help='Whether to train the model?',
                        default=config["train"], action='store_true')

    # #### Training Settings ######
    parser.add_argument('--seed', type=int, default=config["seed"], help='set the seed for reproducibility',
                        required=False)
    parser.add_argument('--n_epochs', type=int, default=config["n_epochs"], required=False)
    parser.add_argument('--epoch_decay', nargs='+', type=int, default=config["epoch_decay"], required=False)
    parser.add_argument('--mu', type=float, default=config["mu"], help='weight decay parameter', required=False)
    parser.add_argument('--momentum', type=float, default=config["momentum"], help='momentum', required=False)
    parser.add_argument('--lr', type=float, default=config["lr"], help='learning rate to use', required=False)
    parser.add_argument('--batch_size', type=int, default=config["batch_size"], help='default is 32', required=False)
    parser.add_argument('--pretrained', default=True, action='store_true', required=False)
    parser.add_argument('--image_size', type=int, default=config["image_size"], required=False)
    parser.add_argument('--crop_size', type=int, default=config["crop_size"], required=False)
    parser.add_argument('--num_workers', type=int, default=config["num_workers"], required=False)
    parser.add_argument('--k', nargs='+', help='value of k for computing the top-k loss and computing top-k accuracy',
                        default=config["k"], type=int, required=False)

    # #### Model Settings #####
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                            'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                            'mobilenet_v2', 'inception_v3', 'alexnet', 'squeezenet',
                                            'shufflenet', 'wide_resnet50_2', 'wide_resnet101_2',
                                            'vgg11', 'mobilenet_v3_large', 'mobilenet_v3_small',
                                            'inception_resnet_v2', 'inception_v4', 'efficientnet_b0',
                                            'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                            'efficientnet_b4', 'vit_base_patch16_224'],
                        default=config["model"], help='choose the model you want to train on', required=False)

    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available(), )
    args = parser.parse_args()
    dict_args = vars(args)

    # ################################# DOWNLOAD DATASET FROM DRIVE ##############################################
    download_data_files(config['download_dir']+"/bioscan_dataset", download=dict_args['download'])

    # ################################# PRINT DATASET STATISTICS #################################################
    show_statistics(data_type=dict_args['data_type'], dataset_name=dict_args['dataset_name'],
                    metadata_dir=f"{dict_args['dataset_dir']}/{dict_args['dataset_name']}/{dict_args['dataset_name']}_metadata.tsv",
                    show=dict_args['print_statistics'])

    # ################################### CREATE DATASET SPLIT ###################################################
    make_split(dict_args)

    # ################################# PRINT DATA SPLIT STATISTICS ##############################################
    show_statistics(data_type=dict_args['data_type'], dataset_name=f"{dict_args['dataset_name']}_train",
                    metadata_dir=f"{dict_args['dataset_dir']}/{dict_args['dataset_name']}/{dict_args['dataset_name']}_train_metadata.tsv",
                    show=dict_args['print_statistics'])

    show_statistics(data_type=dict_args['data_type'], dataset_name=f"{dict_args['dataset_name']}_validation",
                    metadata_dir=f"{dict_args['dataset_dir']}/{dict_args['dataset_name']}/{dict_args['dataset_name']}_validation_metadata.tsv",
                    show=dict_args['print_statistics'])

    show_statistics(data_type=dict_args['data_type'], dataset_name=f"{dict_args['dataset_name']}_test",
                    metadata_dir=f"{dict_args['dataset_dir']}/{dict_args['dataset_name']}/{dict_args['dataset_name']}_test_metadata.tsv",
                    show=dict_args['print_statistics'])

    # ###################################### DATALOADER ##########################################################
    train_dataloader, val_dataloader, test_dataloader, dataset_attributes = get_dataloader(dict_args)

    # ###################################### TRAINING MODEL ######################################################
    train(args, train_dataloader, val_dataloader, dataset_attributes)

    # ###################################### TRAINING MODEL ######################################################
    test(args, test_dataloader, dataset_attributes)








