import datetime
from BioScanDataSet import show_statistics, show_dataset_statistics
from bioscan_datadownload import download_data_files
from bioscan_datasplit import make_split
from bioscan_dataloader import get_dataloader
from train import train
from test import test
from crop_image import run_crop_tool
from configurations import set_configurations
from configurations import make_path_configs


def get_exp_configs():
    """
        This function delivers model configurations ...
        If you want to set an argument from terminal then set required=True for it.

        :return: By the start of a training experiment, a folder named by the data and time <YrMoDa_HrMiSe> is created
        in the result directory which contains configuration .txt file and saved logs of the experiment.
        """

    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    group_levels = {'0': 'domain', '1': 'kingdom', '2': 'phylum', '3': 'class', '4': 'order',
                    '5': 'family', '6': 'subfamily', '7': 'tribe', '8': 'genus', '9': 'species',
                    '10': 'subspecies', '11': 'name'}

    dataset_names = {'1': '200K_dataset', '2': '200K_insect_dataset', '3': '200K_insect_diptera_dataset',
                     '4': 'bioscan_dataset', '5': 'bioscan_insect_dataset', '6': 'bioscan_insect_diptera_dataset',
                     '7': '80K_dataset', '8': '80K_insect_dataset', '9': '80K_insect_diptera_dataset'
                     }

    config = {
        "dataset_dir": "",                # root directory of the dataset, where images and dataframe files are saved
        "hdf5_dir": "",                   # root directory to HDF5 data format
        "image_dir": "",                  # root where images are saved if different from dataset_dir
        "results_dir": "",                # where results are saved (set for evaluation of the trained model)
        "dataset_name": "",               # Name of the dataset, exe., small_dataset, medium_dataset, big_dataset
        "group_level": group_levels['4'],  # Set the Taxonomy group level
        "data_format": "",            # the file format used (exe., hdf5)
        "exp_name": "",
        "crop_image": False,
        "make_split": False,
        "print_statistics": False,
        "print_split_statistics": False,
        "dataloader": False,
        "train": False,
        "test": False,
        "no_transform": False,
        "cropped": True,
        "batch_size": 32,
        "image_size": 256,
        "crop_size": 224,
        "num_workers": 4,
        "seed": 1,
        "n_epochs": 100,
        "epoch_decay": [20, 25],
        "momentum": 0.9,
        "mu": 0.0001,
        "lr": 0.01,
        "k": [1, 3, 5, 10],
        "model": "vit_small_patch16_224",
        "loss": "CE"
    }

    return config, timestamp


if __name__ == '__main__':

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    # Get experiment's specific configurations
    exp_configs, timestamp = get_exp_configs()
    # Get model configurations
    configs = set_configurations(exp_configs)
    # Save model configurations
    configs = make_path_configs(configs, timestamp)


    # ################################# DOWNLOAD DATASET FROM DRIVE ##############################################
    download_data_files(configs['download_dir']+"/bioscan_dataset", download=configs['download'])

    # ##################################### RUN PRE-PROCESSING ###################################################
    run_crop_tool(configs, crop_images=configs['crop_image'])

    # ################################# PRINT DATASET STATISTICS #################################################
    show_dataset_statistics(dataset_name=configs['dataset_name'],
                            metadata_dir=f"{configs['dataset_dir']}/{configs['dataset_name']}/{configs['dataset_name']}_metadata.tsv",
                            show=configs['print_statistics'])

    # ################################### CREATE DATASET SPLIT ###################################################
    data_idx_label = make_split(configs)

    # ################################# PRINT GROUP-LEVEL STATISTICS #############################################
    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'],
                    dataset_name=f"{configs['dataset_name']}",
                    metadata_dir=f"{configs['dataset_dir']}/{configs['dataset_name']}/{configs['dataset_name']}_metadata.tsv",
                    show=configs['print_statistics'])

    # ################################# PRINT DATA SPLIT STATISTICS ##############################################
    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'], dataset_name=f"{configs['dataset_name']}_train",
                    metadata_dir=f"{configs['dataset_dir']}/{configs['dataset_name']}/{configs['dataset_name']}_train_metadata.tsv",
                    show=configs['print_split_statistics'])

    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'], dataset_name=f"{configs['dataset_name']}_validation",
                    metadata_dir=f"{configs['dataset_dir']}/{configs['dataset_name']}/{configs['dataset_name']}_validation_metadata.tsv",
                    show=configs['print_split_statistics'])

    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'], dataset_name=f"{configs['dataset_name']}_test",
                    metadata_dir=f"{configs['dataset_dir']}/{configs['dataset_name']}/{configs['dataset_name']}_test_metadata.tsv",
                    show=configs['print_split_statistics'])

    # ###################################### DATALOADER ##########################################################
    train_dataloader, val_dataloader, test_dataloader, dataset_attributes = get_dataloader(configs, data_idx_label)

    # ###################################### TRAINING MODEL ######################################################
    train(configs, train_dataloader, val_dataloader, dataset_attributes)

    # ###################################### TESTING MODEL ######################################################
    test(configs, test_dataloader, dataset_attributes)








