import datetime
from BioScanDataSet import show_statistics, show_dataset_statistics
from bioscan_datadownload import make_download
from bioscan_datasplit import make_split
from bioscan_dataloader import get_dataloader
from train import train
from test import test
from crop_image import run_crop_tool
from resize_image import make_resize
from configurations import set_configurations, save_configurations


def get_exp_configs():
    """ This function delivers model configurations ..."""

    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    config = {
        "date_time": timestamp,

        # ### Set the following configurations for experiments ### #
        "ID_mapping_path": "dataset/bioscan_1M_insect_dataset_file_ID_mapping.txt",
        "checkpoint_path": None,             # path to the crop_tool checkpoint
        "file_to_download": None,            # file selected to download from the BIOSSCAN_1M_Insect dataset GoogleDrive
        "download_path": None,               # path to download the dataset files
        "dataset_path": None,                # root directory of the dataset files
        "metadata_path": None,               # path to the metadata file (.tsv)
        "hdf5_path": None,                   # path to the HDF5 file containing the original full-size images
        "image_path": None,                  # path to the original full-size images
        "resized_image_path": None,          # path to the resized images
        "resized_hdf5_path": None,           # path to the hdf5 of the resized images
        "cropped_image_path": None,          # path to the cropped images
        "cropped_hdf5_path": None,           # path to the hdf5 file of the cropped images
        "resized_cropped_image_path": None,  # path to the resized cropped images
        "resized_cropped_hdf5_path": None,   # path to the hdf5 file of the resized cropped images
        "results_path": None,                # path to save the results
        "vit_pretrained_path": None,         # path to the directory where the vit backbone is saved

    }
    return config


if __name__ == '__main__':

    # ################################# HYPER_PARAMETER SETTINGS #################################################
    # Get experiment's specific configurations
    exp_configs = get_exp_configs()
    # Get model configurations
    configs = set_configurations(exp_configs)
    # Save model configurations
    configs = save_configurations(configs)

    # ################################# DOWNLOAD DATASET FROM DRIVE ##############################################
    make_download(configs)

    # ################################### PRE-PROCESSING: RESIZING ###############################################
    make_resize(configs["image_path"], configs["resized_image_path"],  configs['resized_hdf5_path'],
                save_binary=True, resize_dimension=256, zip_name="original_256.zip",
                resize_images=configs["resize_image"])

    # ##################################### PRE-PROCESSING: CROPPING #############################################
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

    # ###################################### DATALOADER ###########################################################
    train_dataloader, val_dataloader, test_dataloader, dataset_attributes = get_dataloader(configs, data_idx_label)

    # ###################################### TRAINING MODEL #######################################################
    train(configs, train_dataloader, val_dataloader, dataset_attributes)

    # ###################################### TESTING MODEL ########################################################
    test(configs, test_dataloader, dataset_attributes)








