import datetime
from BioScanDataSet import show_statistics, show_dataset_statistics
from bioscan_datadownload import make_download
from bioscan_datasplit import make_split
from bioscan_dataloader import get_dataloader
from train import train
from test import test
from crop_image import run_crop_tool
from resize_image import make_resize
from configurations import set_configurations, make_path_configs, BioScan_Configurations, get_group_level


def get_exp_configs():
    """
        This function delivers model configurations ...
        If you want to set an argument from terminal then set required=True for it.

        :return: By the start of a training experiment, a folder named by the data and time <YrMoDa_HrMiSe> is created
        in the result directory which contains configuration .txt file and saved logs of the experiment.
        """

    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base_configs = BioScan_Configurations(exp_ID=5)

    config = {
        "date_time": timestamp,
        "experiment_names": base_configs.experiment_names,
        "download_path": "",
        "dataset_path": "",
        "metadata_path": "",
        "dataset_name": "",
        "hdf5_path": "",
        "image_path": "",
        "results_path": "",
        "group_level": "",
        "data_format": "",
        "exp_name": "",
        # ### Train/Test Tool ### #
        "vit_pretrained": "",
        "best_model": "",
        # ### Crop Tool ### #
        "checkpoint_path": "",
    }
    return config


if __name__ == '__main__':

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    # Get experiment's specific configurations
    exp_configs = get_exp_configs()
    # Get model configurations
    configs = set_configurations(exp_configs)
    # Get group_level from experiments name
    configs['group_level'] = get_group_level(exp_name=configs['exp_name'])
    # Save model configurations
    configs = make_path_configs(configs)

    # ################################# DOWNLOAD DATASET FROM DRIVE ##############################################
    make_download(configs)

    # ################################# RESIZE IMAGES ##############################################
    make_resize(configs)

    # ##################################### RUN PRE-PROCESSING ###################################################
    run_crop_tool(configs)

    # ################################# PRINT DATASET STATISTICS #################################################
    show_dataset_statistics(configs)

    # ################################### CREATE DATASET SPLIT ###################################################
    data_idx_label = make_split(configs)

    # ################################# PRINT GROUP-LEVEL STATISTICS #############################################
    show_statistics(configs, gt_ID=data_idx_label, split='all')

    # ################################# PRINT DATA SPLIT STATISTICS ##############################################
    show_statistics(configs, gt_ID=data_idx_label, split='train')

    show_statistics(configs, gt_ID=data_idx_label, split='validation')

    show_statistics(configs, gt_ID=data_idx_label, split='test')

    # ###################################### DATALOADER ##########################################################
    train_dataloader, val_dataloader, test_dataloader, dataset_attributes = get_dataloader(configs, data_idx_label)

    # ###################################### TRAINING MODEL ######################################################
    train(configs, train_dataloader, val_dataloader, dataset_attributes)

    # ###################################### TESTING MODEL ######################################################
    test(configs, test_dataloader, dataset_attributes)








