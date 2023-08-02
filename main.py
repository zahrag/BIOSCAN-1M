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
    """ This function delivers model configurations ..."""

    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base_configs = BioScan_Configurations(exp_ID=5)

    config = {
        "date_time": timestamp,
        "experiment_names": base_configs.experiment_names,

        # ### Set the following configurations for experiments ### #
        "ID_mapping_path": "dataset/bioscan_1M_insect_dataset_file_ID_mapping.txt",
        "file_to_download": None,            # File selected to download from the BIOSSCAN_1M_Insect dataset GoogleDrive
        "download_path": None,               # path to download the dataset files
        "dataset_path": None,                # root directory of the dataset files
        "metadata_path": None,               # path to the metadata file (.tsv)
        "hdf5_path": None,                   # path to the HDF5 file containing the original full-size images
        "image_path": '/home/zahra/Desktop/BioScan/BioScan/dataset/downloaded_drive/images',                  # path to the original full-size images
        "resized_image_path": None,          # path to the resized images
        "resized_hdf5_path": None,           # path to the hdf5 of the resized images
        "cropped_image_path": '/home/zahra/Desktop/BioScan/BioScan/dataset/downloaded_drive/images_cropped',          # path to the cropped images
        "cropped_hdf5_path": None,           # path to the hdf5 file of the cropped images
        "resized_cropped_image_path": None,  # path to the resized cropped images
        "resized_cropped_hdf5_path": None,   # path to the hdf5 file of the resized cropped images
        "results_path": None,                # path to save the results
        "vit_pretrained_path": None,         # path to the directory where the vit backbone is saved
        "checkpoint_path": "/home/zahra/Desktop/BioScan/BioScan/crop_tool/checkpoint_dir/training_with_1000_pinned_and_1000_wells_images.ckpt",             # path to the directory where the crop tool checkpoint is saved

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
                save_binary=True, resize_dimension=256, zip_name="original_256.zip",
                resize_images=configs["resize_image"])

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








