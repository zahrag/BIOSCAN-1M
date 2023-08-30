from torch.utils.data import Dataset
import random
import itertools
import pandas as pd
from BioScanDataSet import BioScan
from utils import make_tsv


class BioScanSplit(Dataset):

    def __init__(self, method="regular", train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2):

       """
       This class handles data splitting  ...
       :param method: which method used to split data?
       :param train_ratio: percentage of all samples used for training set.
       :param validation_ratio: percentage of all samples used for validation set.
       :param test_ratio: percentage of all samples used for test set.
       """

       self.method = method
       self.train_ratio = train_ratio
       self.validation_ratio = validation_ratio
       self.test_ratio = test_ratio

    def make_regular_split(self, data_dict, tr_perc, val_perc, ts_perc):
        """
        This function implements class-based split mechanism.
        :param data_dict: Data dictionary relates sample indexes to class labels.
        :param tr_perc: Ratio used for the training data samples.
        :param val_perc: Ratio used for the validation data samples.
        :param ts_perc: Ratio used for the test data samples.
        :return:
        """

        data_dict_remained = {}
        tr_set = []
        val_set = []
        ts_set = []
        for key in data_dict.keys():
            samples = data_dict[key]
            n_sample = len(samples)
            random.shuffle(samples)
            n_tr_samples = int(round(tr_perc * n_sample))
            n_val_samples = int(round(val_perc * n_sample))
            n_ts_samples = int(round(ts_perc * n_sample))

            if n_sample < 6 and n_sample > 2:
                ts_set.append([samples[0]])
                val_set.append([samples[1]])
                if n_sample == 3:
                    tr_set.append([samples[2]])
                else:
                    tr_set.append(samples[2:])

                data_dict_remained[key] = data_dict[key]

            elif n_tr_samples != 0 and n_val_samples != 0 and n_ts_samples != 0:
                n_diff = n_sample - (n_ts_samples + n_val_samples + n_tr_samples)
                if n_diff != 0:
                    n_tr_samples += n_diff
                tr_set.append(samples[:n_tr_samples])
                val_set.append(samples[n_tr_samples:n_tr_samples + n_val_samples])
                ts_set.append(samples[-n_ts_samples:])

                data_dict_remained[key] = data_dict[key]

            else:
                print(f"Taxa  {key} ---- does NOT have enough samples for data split!")

        train_indexes = list(itertools.chain(*tr_set))
        validation_indexes = list(itertools.chain(*val_set))
        test_indexes = list(itertools.chain(*ts_set))

        # Sanity check: Get missed samples.
        split_all = [train_indexes, validation_indexes, test_indexes]
        split_all = list(itertools.chain(*split_all))
        concatenated_list = [item for sublist in data_dict_remained.values() for item in sublist]
        missed_ids = list(set(concatenated_list) - set(split_all))
        ts_set = [test_indexes, missed_ids]
        test_indexes = list(itertools.chain(*ts_set))

        return data_dict_remained, train_indexes, validation_indexes, test_indexes

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

    def save_split_metadata_separately(self, df, train_indexes, validation_indexes, test_indexes,
                                       group_level='order', dataset_name='', data_dir=None):
        """
            This function saves dataframe files (.tsv) for train, validation and test sets separately.
            """

        train_df, validation_df, test_df = self.get_split_metadata(df, train_indexes, validation_indexes, test_indexes)

        # Save DataFrames (.tsv)
        make_tsv(train_df,
                 name=f"{dataset_name}_{group_level}_train_metadata.tsv",
                 path=f"{data_dir}/{dataset_name}")

        make_tsv(validation_df,
                 name=f"{dataset_name}_{group_level}_validation_metadata.tsv",
                 path=f"{data_dir}/{dataset_name}")

        make_tsv(test_df,
                 name=f"{dataset_name}_{group_level}_test_metadata.tsv",
                 path=f"{data_dir}/{dataset_name}")

    def get_split_dict(self, data_dict, max_num=0):

        """
        This function returns a data dictionary by sampling a parent dictionary applying a
        stratified class-based sampling strategy.
        :param data_dict: Parent data dictionary.
        :param max_num: Maximum number of samples of the child set.
        :return:
        """

        data_dict_child = {}
        for key in list(data_dict.keys()):
            data_dict_child[key] = []

        num = 0
        while num < max_num:
            for key in list(data_dict.keys())[::-1]:
                id_list = data_dict[key]
                next_id = len(data_dict_child[key])
                if next_id < len(id_list):
                    index = id_list[next_id]
                    data_dict_child[key].append(index)
                    num += 1
                else:
                    continue

        return data_dict_child

    def get_diptera_family_data_dict(self, dataset, n_family=40):

        """
        This function generates data dictionary of the order Diptera most populus families.
        :param dataset: Dataset.
        :param n_family: Number of most populus families.
        :return:
        """

        data_list_family = dataset.df['family'].to_list()
        data_list_order = dataset.df['order'].to_list()
        for id, order in enumerate(data_list_order):
            if order != 'Diptera':
                data_list_family[id] = 'not_diptera'

        data_dict = dataset.make_data_dict(data_list_family, dataset.df.index)
        del data_dict['not_diptera']
        if 'not_classified' in data_dict.keys():
            del data_dict['not_classified']

        sorted_families = list(data_dict.keys())
        selected_families = sorted_families[:n_family]
        data_dict_updated = {}
        for family in selected_families:
            data_dict_updated[family] = data_dict[family]

        return data_dict_updated

    def get_subset_dict(self, dataset, exp='', max_num=200, group_level='', split=''):

        """
        This function creates data dictionary of subset.
        :param dataset: Dataset class.
        :param exp: Experiments name.
        :param max_num: Maximum number of samples of the subset.
        :param group_level: Taxonomy Group-Level.
        :param split: Split: train, validation, test.
        :return: Data list, class to id.
        """

        if split == 'train':
            factor = self.train_ratio
        elif split == 'validation':
            factor = self.validation_ratio
        elif split == 'test':
            factor = self.test_ratio
        else:
            print("Please set a split: train, or validation, or test!")
            return

        name = ''.join(list(exp)[-6:])
        if name == "family" and group_level == "family":
            parent_exp_name = 'large_diptera_family'
        elif name == "_order" and group_level == "order":
            parent_exp_name = 'large_insect_order'
        else:
            print("Set the experiments name and group-level correctly!")
            return

        if parent_exp_name not in dataset.df_categories:
            print(f"First split parent set:{parent_exp_name}")
            return

        # Create data dictionary of the split
        data_list = dataset.df[group_level].to_list()
        split_list = dataset.df[parent_exp_name].to_list()
        for id, sp in enumerate(split_list):
            if sp != split:
                data_list[id] = "no_split"

        # Get data dictionary of the split set
        data_dict = dataset.make_data_dict(data_list, dataset.df.index)
        del data_dict['no_split']
        data_idx_label = dataset.class_to_ids(data_dict)

        # Stratify class-based subset sampling
        data_dict = self.get_split_dict(data_dict, max_num=int(factor*max_num))
        concatenated_list = [item for sublist in data_dict.values() for item in sublist]
        if len(concatenated_list) > (factor*max_num):
            top_key = data_dict[list(data_dict.keys())[0]]
            data_dict[list(data_dict.keys())[0]] = top_key[:-int(len(concatenated_list)-(factor*max_num))]
            concatenated_list = [item for sublist in data_dict.values() for item in sublist]

        return concatenated_list, data_idx_label

    def get_order_split_ids(self, dataset, group_level=''):
        """
        This function creates train, validation and test ids of large_insect_order dataset using split of
        large_diptera_family dataset.
        :param dataset: Dataset class.
        :param group_level: Taxonomy Group-Level.
        :return: insect_order train, validation and test split indices.
        """

        data_list = dataset.df[group_level].to_list()

        # Get Diptera Family-Level Split list
        split_list = dataset.df[dataset.experiment_names[0]].to_list()

        tr_ids = [id for id, sp in enumerate(split_list) if sp == 'train']
        val_ids = [id for id, sp in enumerate(split_list) if sp == 'validation']
        ts_ids = [id for id, sp in enumerate(split_list) if sp == 'test']

        data_list_remained = ['no_split' for id in dataset.df.index]
        for id, sp in enumerate(split_list):
            if sp == 'no_split':
                data_list_remained[id] = data_list[id]

        data_dict = dataset.make_data_dict(data_list_remained, dataset.df.index)
        del data_dict['no_split']

        data_dict_remained, tr_ids_r, val_ids_r, ts_ids_r = self.get_split_ids(data_dict)

        tr_set = [tr_ids, tr_ids_r]
        train_indexes = list(itertools.chain(*tr_set))
        val_set = [val_ids, val_ids_r]
        validation_indexes = list(itertools.chain(*val_set))
        ts_set = [ts_ids, ts_ids_r]
        test_indexes = list(itertools.chain(*ts_set))

        return train_indexes, validation_indexes, test_indexes

    def save_split_metadata(self, dataset, tr_indexes, val_indexes, ts_indexes,
                            exp_name="", metadata_name="", metadata_path=""):
        """
        This function saves experiments split list as columns of metadata file.
        :param dataset: Dataset Class.
        :param tr_indexes: Train indices.
        :param val_indexes: Validation Indices.
        :param ts_indexes: Test Indices.
        :param exp_name: Experiment name.
        :param metadata_name: Name of metadata to save.
        :param metadata_path: Path to the metadata file to save in.
        :return:
        """

        df = dataset.df
        column = ['no_split' for id in df.index]
        for id in tr_indexes:
            column[id] = 'train'
        for id in val_indexes:
            column[id] = 'validation'
        for id in ts_indexes:
            column[id] = 'test'

        if exp_name == 'large_diptera_family':
            updated_df = df.assign(large_diptera_family=column)
        elif exp_name == 'medium_diptera_family':
            updated_df = df.assign(medium_diptera_family=column)
        elif exp_name == 'small_diptera_family':
            updated_df = df.assign(small_diptera_family=column)
        elif exp_name == 'large_insect_order':
            updated_df = df.assign(large_insect_order=column)
        elif exp_name == 'medium_insect_order':
            updated_df = df.assign(medium_insect_order=column)
        elif exp_name == 'small_insect_order':
            updated_df = df.assign(small_insect_order=column)
        else:
            print(f"Not a valid experiment is set:{exp_name}")
            return

        updated_df = pd.DataFrame(updated_df)
        updated_df.reset_index(inplace=True, drop=True)
        make_tsv(updated_df, name=metadata_name, path=metadata_path)


def make_split(configs):
    """
    This function samples and splits BIOSCAN-1M Insect Dataset for 6 experiments conducted in paper.
    NOTE-1: First sample and split BIOSCAN-Diptera dataset!
    NOTE-2: Split parent sets (large_insect_order, large_diptera_family) before sampling and splitting
            their children sets (medium, small)!
    :param configs: Configurations.
    :return:
    """

    dataset = BioScan()
    data_split = BioScanSplit()

    if not configs['make_split']:
        dataset.set_statistics(configs, split='train')
        return dataset.data_idx_label

    # Get data statistics for the whole dataset
    dataset.set_statistics(configs, split='all')
    exp = configs['exp_name']
    g_level = configs['group_level']
    max_num = configs['max_num_sample']

    if exp == 'large_diptera_family' and g_level == 'family':
        data_dict_family = data_split.get_diptera_family_data_dict(dataset)
        data_dict_remained, tr_indexes, val_indexes, ts_indexes = data_split.get_split_ids(data_dict_family)
        data_idx_label = dataset.class_to_ids(data_dict_remained)

    elif exp == 'large_insect_order' and g_level == 'order':
        if 'large_diptera_family' in dataset.df_categories:
            tr_indexes, val_indexes, ts_indexes = data_split.get_order_split_ids(dataset, group_level=g_level)
            data_idx_label = dataset.class_to_ids(dataset.data_dict)
        else:
            print("First split Diptera Family-Level!")
            return

    elif exp in ['medium_diptera_family', 'small_diptera_family', 'medium_insect_order', 'small_insect_order']:

        tr_indexes, data_idx_label = data_split.get_subset_dict(dataset,
                                                                exp=exp,
                                                                max_num=max_num,
                                                                group_level=g_level, split='train')

        val_indexes, data_idx_label = data_split.get_subset_dict(dataset,
                                                                 exp=exp,
                                                                 max_num=max_num,
                                                                 group_level=g_level, split='validation')

        ts_indexes, data_idx_label = data_split.get_subset_dict(dataset,
                                                                exp=exp,
                                                                max_num=max_num,
                                                                group_level=g_level, split='test')
    else:
        print("Set experiments name and group level correctly!")
        return

    data_split.save_split_metadata(dataset, tr_indexes, val_indexes, ts_indexes,
                                   exp_name=configs['exp_name'],
                                   metadata_name=f'BIOSCAN_Insect_Dataset_metadata.tsv',
                                   metadata_path=f"{configs['dataset_path']}/")

    return data_idx_label





