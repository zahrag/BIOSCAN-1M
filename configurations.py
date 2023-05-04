
import argparse
import os
from utils import make_directory
import torch


def set_configurations(config=None):

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    # #### Path Settings ######
    parser.add_argument('--exp_name', type=str, default=config["exp_name"], help='name of the experiment',
                        required=False)
    parser.add_argument('--dataset_name', type=str, default=config["dataset_name"], help='name of the dataset',
                        required=False)
    parser.add_argument('--group_level', type=str, default=config["group_level"], help='type of the data',
                        required=False)
    parser.add_argument('--data_format', type=str, default=config["data_format"], help='format of the dataset',
                        required=False)
    parser.add_argument('--log', type=str, default="runs", help='Path to the log file', required=False)
    parser.add_argument('--download_dir', type=str, help='Directory to download our dataset',
                        default='', required=False)
    parser.add_argument('--dataset_dir', type=str, help='Directory to our dataset', default=config["dataset_dir"],
                        required=False)
    parser.add_argument('--image_dir', type=str, help='Directory to our images if different from dataset_dir',
                        default=config["image_dir"], required=False)
    parser.add_argument('--hdf5_dir', type=str, help='Directory to our hdf5 data format.', default=config["hdf5_dir"],
                        required=False)
    parser.add_argument('--results_dir', type=str, help='Directory to save results', default=config["results_dir"],
                        required=False)
    parser.add_argument('--no_transform', default=config["no_transform"], action='store_true', required=False)
    parser.add_argument('--cropped', default=config["cropped"], action='store_true', required=False)

    # #### Condition Settings #####
    parser.add_argument('--download', help='Whether to download from drive?',
                        default=False, action='store_true')
    parser.add_argument('--make_split', help='Whether to split dataset into train, validation and test sets?',
                        default=config["make_split"], action='store_true')
    parser.add_argument('--print_statistics', help='Whether to print dataset statistics?',
                        default=config["print_statistics"], action='store_true')
    parser.add_argument('--print_split_statistics', help='Whether to print dataset statistics?',
                        default=config["print_split_statistics"], action='store_true')
    parser.add_argument('--loader', help='Whether to generate dataloader?',
                        default=config["dataloader"], action='store_true')
    parser.add_argument('--train', help='Whether to train the model?',
                        default=config["train"], action='store_true')
    parser.add_argument('--test', help='Whether to test the model?',
                        default=config["test"], action='store_true')
    parser.add_argument('--crop_image', help='Whether to crop dataset images?',
                        default=config["crop_image"], action='store_true')

    # #### Preprocessing: Cropping Settings ######
    parser.add_argument('--meta_path', type=str, default="", help="path to the meta directory")
    parser.add_argument('--image_hdf5', type=str, default="", help="path to the image hdf5y")
    parser.add_argument('--output_dir', type=str, default="", help="path to the image directory")
    parser.add_argument('--output_hdf5', type=str, default="", help="path to the image hdf5y")
    parser.add_argument('--checkpoint_path', type=str, default="", help="Path to the checkpoint.")
    parser.add_argument('--crop_ratio', type=float, default=1.4, help="Scale the bbox to crop larger or small area.")

    # #### Training Settings ######
    parser.add_argument('--seed', type=int, default=config["seed"], help='set the seed for reproducibility', required=False)
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

    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available(),)
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


def make_path_configs(config, timestamp):
    if config["train"]:
        save_dir = os.path.join(os.getcwd(), config["results_dir"])
        save_dir += "/{timestamp:s}_{dataset:s}_epoch{epoch:d}/".format(timestamp=timestamp,
                                                                        dataset=config['dataset_name'],
                                                                        epoch=config["n_epochs"])
        make_directory(save_dir)
        save_configs(timestamp, config, log_dir=save_dir)
        config["results_dir"] = save_dir
        config["exp_name"] = config["dataset_name"]

    return config


