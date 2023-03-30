import os
import wget
import sys


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_data_files(download_folder, download=False):
    """
        This function downloads all files related to the BioScan-1M dataset, including RGB images, metadata and Barcodes.
        Due to the large size of the data files, it is recommended to download in tmp directory exe., scratch directory
        in Compute Canada.
        ulr links of the data files in Google Drive needs be saved in this function.
        :param download_folder: Destination folder where the dataset files are stored.
        :return:
        """

    if not download:
        return

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_web_link_metadata = ""
    data_file_1 = ""
    data_file_2 = ""
    data_file_3 = ""
    data_file_4 = ""
    data_file_5 = ""
    data_file_6 = ""
    data_file_7 = ""
    data_file_8 = ""
    data_file_9 = ""
    data_file_10 = ""

    data_file_weblinks = [download_web_link_metadata, data_file_1, data_file_2, data_file_3, data_file_4, data_file_5,
                          data_file_6, data_file_7, data_file_8, data_file_9, data_file_10, ]

    for cnt in range(len(data_file_weblinks)):
        print(f"\nData File {cnt + 1}")
        wget.download(data_file_weblinks[cnt], out=download_folder, bar=bar_progress)


if __name__ == '__main__':

    root = ""
    download_folder = f"{root}/bioscan_dataset/"
    download_data_files(download_folder)
