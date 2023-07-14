
import os
import wget
from utils import make_directory
import itertools
import gdown


def read_id_mapping(id_mapping_path=""):
    """ Read ID-Mapping-File """
    if not os.path.isfile(id_mapping_path):
        print(f"ID Mapping File is not available to read in:\n{id_mapping_path}")
        return
    file_id_mapping = {}
    with open(id_mapping_path) as fp:
        for i, line in enumerate(fp):
            a_string = line.strip()
            info = [word for word in a_string]
            sep_index = info.index(':')
            file_id_mapping[''.join(info[:sep_index])] = ''.join(info[sep_index + 1:])
    return file_id_mapping


def gdown_download(file_id, file_name, download_path=""):
    """ Download file from drive using gdown """
    make_directory(download_path)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, f"{download_path}/{file_name}")


def make_download(configs):
    """
    This function downloads the files of the BIOSCAN_1M_Insect dataset from GoogleDrive.
    :param configs: Configurations.
    :return:
    """

    if configs['file_to_download'] is None:
        print("No file is selected to download!")
        return

    id_mapping_path = configs['ID_mapping_path']
    download_path = configs['download_path']
    file_selected = configs["file_to_download"]

    parent_folder_main = "1ft17GpcC_xDhx5lOBhYAEtox0sdZchjA"
    parent_folder_original = "15CIT4DIe_--I20h-KwLbFmGefpBMHx5A"
    parent_folder_cropped = "1CUBZLO5u_uEYptZE-S8B6RVeBNc5jFdx"

    dataset_metadata_tsv = ['BIOSCAN_Insect_Dataset_metadata.tsv']
    dataset_metadata_jsonld = ['BIOSCAN_Insect_Dataset_metadata.jsonld']
    dataset_original_256_zip = ['original_256.zip']
    dataset_cropped_256_zip = ['cropped_256.zip']
    dataset_original_256_hdf5 = ['original_256.hdf5']
    dataset_cropped_256_hdf5 = ['original_256.hdf5']
    dataset_original_package = [f"bioscan_images_original_full_part{id + 1}.zip" for id in range(113)]
    dataset_cropped_package = [f"bioscan_images_cropped_full_part{id + 1}.zip" for id in range(113)]
    files_list = [dataset_metadata_tsv, dataset_metadata_jsonld, dataset_original_256_zip,
                  dataset_cropped_256_zip, dataset_original_256_hdf5, dataset_cropped_256_hdf5,
                  dataset_original_package, dataset_cropped_package]
    files_list = list(itertools.chain(*files_list))

    if file_selected not in files_list:
        raise RuntimeError(f"File:{file_selected} is not available for download!")

    file_id_mapping = read_id_mapping(id_mapping_path=id_mapping_path)

    if file_selected in dataset_original_package:
        parent_folder_id = parent_folder_original

    elif file_selected in dataset_cropped_package:
        parent_folder_id = parent_folder_cropped

    else:
        parent_folder_id = parent_folder_main

    gdown_download(file_id_mapping[file_selected], file_selected, download_path=download_path)




