import datetime
from BioScanDataSet import show_statistics, show_dataset_statistics
from bioscan_datadownload import make_download
from bioscan_datasplit import make_split
from bioscan_dataloader import get_dataloader
from train import train
from test import test
from crop_image import run_crop_tool
from resize_image import make_resize
from configurations import set_configurations, make_path_configs, BioScan_Configurations, get_group_level, extract_package


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
        "dataset_name": "BioScan_Insect_Dataset",

        # ### Set the following configurations for experiments ### #
        "download_path": "path to download the dataset files",
        "dataset_path": "root directory to the dataset files",
        "metadata_path": "path to the metadata file (.tsv)",
        "hdf5_path": "path to the HDF5 file containing the original full-size images",
        "image_path": "path to the original full-size images",
        "results_path": "path to save the results",
        "vit_pretrained": "path to the directory where the vit backbone is saved",
        "checkpoint_path": "path to the directory where the crop tool checkpoint is saved",
        "data_format": "data format used for experiments",
        "exp_name": "experiment name",
        "best_model": "best model selected from validation results",

    }
    return config


if __name__ == '__main__':

    # ################################# HYPER_PARAMETER SETTINGS ##############################################
    # Get experiment's specific configurations
    exp_configs = get_exp_configs()
    # Get model configurations
    configs = set_configurations(exp_configs)
    # Extract input package if compressed
    configs['image_path'] = extract_package(configs['image_path'], data_format=configs['data_format'])
    # Get group_level from experiments name
    configs['group_level'] = get_group_level(exp_name=configs['exp_name'])
    # Save model configurations
    configs = make_path_configs(configs)

    # ################################# DOWNLOAD DATASET FROM DRIVE ##############################################
    make_download(configs)

    # ################################# RESIZE IMAGES ##############################################
    make_resize(configs["image_path"], configs["resized_image_path"],  configs['resized_hdf5_path'],
                saved_as_binary_data=True, resize_dimension=256, zip_name="original_256.zip")

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








