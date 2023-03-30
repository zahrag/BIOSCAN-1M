from torch.utils.data import Dataset
import random
import itertools
import os
import shutil
import pandas as pd
from BioScanDataSet import BioScan
from utils import extract_tar, move_to_dir, make_tar, make_tsv, copy_to_dir


class BioScanSplit(Dataset):

    def __init__(self, method="regular", train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2):

       """
            This class handles data splitting  ...
           :param method: which method used to split data?
           :param train_ratio: percentage of all samples used for training set
           :param validation_ratio: percentage of all samples used for validation set
           :param test_ratio: percentage of all samples used for test set
           """

       self.method = method
       self.train_ratio = train_ratio
       self.validation_ratio = validation_ratio
       self.test_ratio = test_ratio

    def get_split_ids(self, data_dict):
        """
            This function splits data into train, validation and test sets.
            The split is generated per class to address class imbalances,
            which is based on ratios preset for train, validation and test sets.
            """

        print(f"\n\n\t\t\t\t\t\t Data Split \t\t\t\t\t\t")
        tr_perc = self.train_ratio
        val_perc = self.validation_ratio
        ts_perc = self.test_ratio

        data_dict_remained, tr_indexes, val_indexes, ts_indexes = self.make_regular_split(data_dict,
                                                                                          tr_perc, val_perc, ts_perc)

        return data_dict_remained, tr_indexes, val_indexes, ts_indexes

    def make_regular_split(self, data_dict, tr_perc, val_perc, ts_perc):

        data_dict_remained = {}
        tr_set = []
        val_set = []
        ts_set = []
        for cnt, key in enumerate(data_dict.keys()):
            n_sample = len(data_dict[key])
            samples = data_dict[key]
            random.shuffle(samples)
            n_tr_samples = int(tr_perc * n_sample)
            n_val_samples = int(val_perc * n_sample)
            n_ts_samples = int(ts_perc * n_sample)

            if n_tr_samples != 0 and n_val_samples != 0 and n_ts_samples != 0:
                tr_set.append(samples[:n_tr_samples])
                val_set.append(samples[n_tr_samples:n_tr_samples + n_val_samples])
                ts_set.append(samples[-n_ts_samples:])

                data_dict_remained[key] = data_dict[key]

            else:
                print(f"Order  {key} ---- NOT enough samples for data split!")

        train_indexes = list(itertools.chain(*tr_set))
        validation_indexes = list(itertools.chain(*val_set))
        test_indexes = list(itertools.chain(*ts_set))

        return data_dict_remained, train_indexes, validation_indexes, test_indexes

    def get_split_images(self, image_list, train_indexes, validation_indexes, test_indexes):

        """
            This function creates lists of image names for train, validation and test sets.
            """

        train_images = [image_list[id] for id in train_indexes]
        validation_images = [image_list[id] for id in validation_indexes]
        test_images = [image_list[id] for id in test_indexes]

        return train_images, validation_images, test_images

    def get_split_metadata(self, df, train_indexes, validation_indexes, test_indexes):

        """
           This function creates dataframe files for train, validation and test sets.
           """

        train_df = [df.iloc[id] for id in train_indexes]
        train_df = pd.DataFrame(train_df)
        train_df.reset_index(inplace=True, drop=True)

        validation_df = [df.iloc[id] for id in validation_indexes]
        validation_df = pd.DataFrame(validation_df)
        validation_df.reset_index(inplace=True, drop=True)

        test_df = [df.iloc[id] for id in test_indexes]
        test_df = pd.DataFrame(test_df)
        test_df.reset_index(inplace=True, drop=True)

        return train_df, validation_df, test_df

    def save_split_metadata(self, df, train_indexes, validation_indexes, test_indexes,
                            dataset_name="small_dataset", data_dir=None):
        """
            This function saves dataframe files (.tsv) for train, validation and test sets.
            """

        train_df, validation_df, test_df = self.get_split_metadata(df, train_indexes, validation_indexes, test_indexes)

        # Save DataFrames (.tsv)
        make_tsv(train_df, name=f"{dataset_name}_train_metadata.tsv", path=f"{data_dir}/{dataset_name}/")
        make_tsv(validation_df, name=f"{dataset_name}_validation_metadata.tsv", path=f"{data_dir}/{dataset_name}/")
        make_tsv(test_df, name=f"{dataset_name}_test_metadata.tsv", path=f"{data_dir}/{dataset_name}/")

    def save_images(self, image_tar, image_list, train_indexes, validation_indexes, test_indexes,
                    dataset_name="small_dataset", data_dir=None, save_split_images=False):

        """
            This function saves images (.jpg) for train, validation and test sets if requested.
            """

        if not save_split_images:
            return

        train_images, validation_images, test_images = self.get_split_images(image_list,
                                                                             train_indexes, validation_indexes,
                                                                             test_indexes)

        print("\nSet split directories to save train, validation and test images ...\n")
        train_dir = f"{data_dir}/{dataset_name}/{dataset_name}_train_images"
        validation_dir = f"{data_dir}/{dataset_name}/{dataset_name}_validation_images"
        test_dir = f"{data_dir}/{dataset_name}/{dataset_name}_test_images"

        image_tar_names = []
        cnt = 0
        for tar in image_tar:
            if tar not in image_tar_names:
                image_tar_names.append(tar)
                cnt += 1

                print("\nSet temporary directory to store images ...\n")
                tmp_path = f"{data_dir}/tmp/"

                print(f"Image-Tar-File-{cnt}: {tar} is extracting ...")
                extract_tar(tar_file=f'{data_dir}/{tar}', path=tmp_path)
                imgs = [image for image in os.listdir(tmp_path)]

                not_sorted = 0
                for id, img in enumerate(imgs):
                    if img in train_images:
                        print(f"Train ---- {img} found in {tar} ----")
                        copy_to_dir(source=f"{tmp_path}{img}", destination=train_dir)

                    elif img in validation_images:
                        print(f"Validation ---- {img} found in {tar} ----")
                        copy_to_dir(source=f"{tmp_path}{img}", destination=validation_dir)

                    elif img in test_images:
                        print(f"Test ---- {img} found in {tar} ----")
                        copy_to_dir(source=f"{tmp_path}{img}", destination=test_dir)

                    else:
                        not_sorted += 1
                        # print(f"Image {img} is not sorted in any split sets! ")

                print(f"\n Total of {not_sorted} images are not in any split set")
                print("\n Remove temporary path ...")
                shutil.rmtree(tmp_path)

        print("\nCreate tar folder of Train set ...")
        make_tar(name=f"{dataset_name}_train_images.tar", path=train_dir)

        print("\nCreate tar folder of Validation set ...")
        make_tar(name=f"{dataset_name}_validation_images.tar", path=validation_dir)

        print("\nCreate tar folder of Test set ...")
        make_tar(name=f"{dataset_name}_test_images.tar", path=test_images)


def make_split(args):

    if not args['make_split']:
        return

    dataset = BioScan()
    data_split = BioScanSplit()

    # Get data statistics for the whole dataset
    dataset.set_statistics(data_type=args['data_type'],
                           metadata_dir=f"{args['dataset_dir']}/{args['dataset_name']}/{args['dataset_name']}_metadata.tsv")

    # Split the whole dataset into Train, Validation and Test sets: Get indexes
    data_dict_remained, tr_indexes, val_indexes, ts_indexes = data_split.get_split_ids(dataset.data_dict)

    # Split the whole dataset into Train, Validation and Test sets: Save RGB images
    data_split.save_images(dataset.image_tar, dataset.image_names, tr_indexes, val_indexes, ts_indexes,
                           dataset_name=args['dataset_name'], data_dir=args['dataset_dir'],
                           save_split_images=False)

    # Split the whole dataset into Train, Validation and Test sets: Save Split Metadata (.tsv)
    data_split.save_split_metadata(dataset.df, tr_indexes, val_indexes, ts_indexes,
                                   dataset_name=args['dataset_name'], data_dir=args['dataset_dir'])





