import datetime
from BioScanDataSet import show_statistics, show_dataset_statistics
from bioscan_datadownload import download_data_files
from bioscan_datasplit import make_split
from bioscan_dataloader import get_dataloader
from train import train
from test import test
from crop_image import run_crop_tool
from configurations import set_configurations, make_path_configs


def get_exp_configs():
    """
        This function delivers model configurations ...
        If you want to set an argument from terminal then set required=True for it.

        :return: By the start of a training experiment, a folder named by the data and time <YrMoDa_HrMiSe> is created
        in the result directory which contains configuration .txt file and saved logs of the experiment.
        """

    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    config = {
        "date_time": timestamp,
        "dataset_path": "",
        "metadata_path": "",
        "dataset_name": "",
        "hdf5_path": "",
        "image_path": "",
        "results_path": "",
        "exp_name": "",
    }

    return config


if __name__ == '__main__':

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    # Get experiment's specific configurations
    exp_configs = get_exp_configs()
    # Get model's configurations
    configs = set_configurations(exp_configs)
    # Save model's configurations
    configs = make_path_configs(configs, exp_configs["date_time"])

    # ################################# DOWNLOAD DATASET FROM DRIVE ##############################################
    download_data_files(f"{configs['download_path']}/bioscan_dataset", download=configs['download'])

    # ##################################### RUN PRE-PROCESSING ###################################################
    run_crop_tool(configs, crop_images=configs['crop_image'])

    # ################################# PRINT DATASET STATISTICS #################################################
    show_dataset_statistics(dataset_name=configs['dataset_name'],
                            metadata_dir=configs['metadata_path'],
                            show=configs['print_statistics'])

    # ################################### CREATE DATASET SPLIT ###################################################
    data_idx_label = make_split(configs)

    # ################################# PRINT GROUP-LEVEL STATISTICS #############################################
    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'],
                    dataset_name=f"{configs['dataset_name']}",
                    metadata_dir=configs['metadata_path'],
                    show=configs['print_split_statistics'])

    # ################################# PRINT DATA SPLIT STATISTICS ##############################################
    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'],
                    dataset_name=f"{configs['dataset_name']} Train Set",
                    metadata_dir=configs['metadata_path'],
                    show=configs['print_split_statistics'])

    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'],
                    dataset_name=f"{configs['dataset_name']} Validation Set",
                    metadata_dir=configs['metadata_path'],
                    show=configs['print_split_statistics'])

    show_statistics(gt_ID=data_idx_label, group_level=configs['group_level'],
                    dataset_name=f"{configs['dataset_name']} Test Set",
                    metadata_dir=configs['metadata_path'],
                    show=configs['print_split_statistics'])

    # ###################################### DATALOADER ##########################################################
    train_dataloader, val_dataloader, test_dataloader, dataset_attributes = get_dataloader(configs, data_idx_label)

    # ###################################### TRAINING MODEL ######################################################
    train(configs, train_dataloader, val_dataloader, dataset_attributes)

    # ###################################### TESTING MODEL ######################################################
    test(configs, test_dataloader, dataset_attributes)








