
import pandas as pd
from torch.utils.data import Dataset
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import make_directory, read_tsv


class BioScan(Dataset):
    def __init__(self):
        """
            This class handles getting, setting and showing data statistics ...
            """

    def get_statistics(self, experiment_names, metadata_dir, split='', exp='', make_split=False):
        """
        This function sets data attributes read from metadata file of the dataset.
        This includes biological taxonomy annotations, DNA barcode indexes and RGB image names and chunk numbers.
        :param experiment_names: Name of 6 experiments conducted in BIOSCAN-1M Insect paper.
        :param metadata_dir: Path to the Metadata file (.tsv)
        :param split: Set split: all, train, validation, test
        :param exp: Experiment Name.
        :param make_split: If splitting dataset?
        :return:
        """

        # Get experiments name
        self.experiment_names = experiment_names

        self.metadata_dir = metadata_dir
        self.df = self.read_metadata(metadata_dir, split, exp, make_split=make_split)
        self.index = self.df.index.to_list()
        self.df_categories = self.df.keys().to_list()
        self.n_DatasetAttributes = len(self.df_categories)

        # Biological Taxonomy
        self.taxa_gt_sorted = {'0': 'domain', '1': 'kingdom', '2': 'phylum', '3': 'class', '4': 'order',
                               '5': 'family', '6': 'subfamily', '7': 'tribe', '8': 'genus', '9': 'species',
                               '10': 'subspecies', '11': 'name'}

        self.taxonomy_groups_list_dict = {}
        for taxa in self.taxa_gt_sorted.values():
            if taxa in self.df_categories:
                self.taxonomy_groups_list_dict[taxa] = self.df[taxa].to_list()

        # Barcode and data Indexing
        self.barcode_indexes = ['nucraw', 'uri', 'processid', 'sampleid']

        self.barcode_list_dict = {}
        for bar in self.barcode_indexes:
            if bar in self.df_categories:
                self.barcode_list_dict[bar] = self.df[bar].to_list()

        # RGB Images
        self.image_names = self.df['image_file'].to_list()

        # Data Chunk index
        self.chunk_length = 10000
        self.chunk_index = self.df['chunk_number'].to_list()

    def __len__(self):
        return len(self.index)

    def read_metadata(self, metadata_dir, split, exp, make_split=False):
        """
        This function reads .tsv type metadata file.
        :param metadata_dir: Path to the metadata file.
        :param split: Set split including all, train, validation and test.
        :param exp: Experiment name.
        :param make_split: If making split?
        :return: Dataframe metadata.
        """

        if os.path.isfile(metadata_dir) and os.path.splitext(metadata_dir)[1] == '.tsv':
            df = read_tsv(metadata_dir)
        else:
            print(f"Not a CVS metadata file exits in the directory:\n{metadata_dir}")
            return

        if make_split:
            return df

        elif exp in df.columns:

            if split == 'all':
                df_split = [df.iloc[id] for id, cl in enumerate(df[exp]) if cl != 'no_split']
            else:
                df_split = [df.iloc[id] for id, cl in enumerate(df[exp]) if cl == split]

            df_split = pd.DataFrame(df_split)
            df_split.reset_index(inplace=True, drop=True)
            return df_split

        else:
            print(f"Experiment split is not available:{exp}!")
            return

    def set_statistics(self, configs, split=''):
        """
        This function sets dataset statistics.
        :param configs: Configurations.
        :param split: Split: all, train, validation, test.
        :return:
        """

        self.get_statistics(configs['experiment_names'], configs["metadata_path"],
                            exp=configs["exp_name"],
                            make_split=configs["make_split"],
                            split=split)

        # Get data list as one of the Biological Taxonomy
        if configs["group_level"] in self.taxonomy_groups_list_dict.keys():
            self.data_list = self.taxonomy_groups_list_dict[configs["group_level"]]

        else:
            print(f'Dataset Does NOT Have the Taxonomy Group Ranking: {configs["group_level"]}')
            return

        # Get the data dictionary
        self.data_dict = self.make_data_dict(self.data_list, self.index)

        # Get numbered labels of the classes
        self.data_idx_label = self.class_to_ids(self.data_dict)

        # Get number of samples per class
        self.n_sample_per_class = self.get_n_sample_class(self.data_dict)

        # Get numbered data samples list
        self.data_list_ids = self.class_list_idx(self.data_list, self.data_idx_label)

    def make_data_dict(self, data_list, index):
        """
        This function creates data dictionary key:label(exe., order name), value:indexes in data list.
        :return:
        """

        data_dict = {}
        for cnt, data in enumerate(data_list):
            if not isinstance(data, str):
                if math.isnan(data):
                    data_list[cnt] = "not_classified"
                else:
                    print("Not a string type data name is detected!")
                    return

        data_names = []
        for data in data_list:
            if data not in data_names:
                data_names.append(data)

        for name in data_names:
            indexes = [ind for ind in index if data_list[index.index(ind)] == name]
            data_dict[name] = indexes

        n_sample_per_class = [len(class_samples) for class_samples in list(data_dict.values())]
        indexed_list = list(enumerate(n_sample_per_class))
        sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
        original_indices_sorted = [x[0] for x in sorted_list]

        class_names = list(data_dict.keys())
        sorted_class_names = [class_names[ind] for ind in original_indices_sorted]

        sorted_data_dict = {}
        for name in sorted_class_names:
            sorted_data_dict[name] = data_dict[name]

        return sorted_data_dict

    def class_to_ids(self, data_dict):

        """
        This function creates a numeric id for a class.
        :param data_dict: Data dictionary corresponding each class to its sample ids
        :return:
        """
        data_idx_label = {}
        data_names = list(data_dict.keys())
        for name in data_names:
            data_idx_label[name] = data_names.index(name)

        return data_idx_label

    def get_n_sample_class(self, data_dict):
        """
        This function computes total number of samples per class.
        :param data_dict: Data dictionary corresponding each class to its sample ids
        :return:
        """

        data_samples_list = list(data_dict.values())
        n_sample_per_class = [len(class_samples) for class_samples in data_samples_list]

        return n_sample_per_class

    def class_list_idx(self, data_list, data_idx_label):
        """
        This function creates data list of numbered labels.
        :param data_list: data list of class names.
        :param data_idx_label: numeric ids of class names
        :return:
        """

        data_list_ids = []
        for data in data_list:
            data_list_ids.append(data_idx_label[data])

        return data_list_ids


def show_dataset_statistics(configs):
    """
    This function shows data statistics from metadata file of the dataset.
    :param configs: Configurations.
    :return:
    """
    if not configs["print_statistics"]:
        return

    print("\n\nCreating data statistics ...")
    print("ATTENTION:This process can take time especially if the dataset is big!")

    dataset = BioScan()
    dataset.get_statistics(configs['experiment_names'], configs["metadata_path"], exp=configs["exp_name"], split='all')

    print("\n\n----------------------------------------------------------------------------------------")
    print(f"\t\t\t\t\t\t\t\tCopyright")
    print("----------------------------------------------------------------------------------------")
    print("Copyright Holder: CBG Photography Group")
    print("Copyright Institution: Centre for Biodiversity Genomics (email:CBGImaging@gmail.com)")
    print("Photographer: CBG Robotic Imager")
    print("Copyright License: Creative Commons-Attribution Non-Commercial Share-Alike (CC BY-NC-SA 4.0)")
    print("Copyright Contact: collectionsBIO@gmail.com")
    print("Copyright Year: 2021")
    print("----------------------------------------------------------------------------------------")

    # Get taxonomy ranking statistics
    dataset_taxa = [taxa for taxa in dataset.taxa_gt_sorted.values() if taxa in dataset.df_categories]

    # Get subgroups statistics
    set_group_level = configs["group_level"]
    group_level_dict = {}
    for taxa in dataset_taxa:
        configs["group_level"] = taxa
        dataset.set_statistics(configs, split='all')
        group_level_dict[f"{taxa}_n_subgroups"] = len(dataset.data_dict)
        group_level_dict[f"{taxa}_n_not_grouped_samples"] = 0
        if "not_classified" in dataset.data_dict:
            group_level_dict[f"{taxa}_n_not_grouped_samples"] = len(dataset.data_dict["not_classified"])
            group_level_dict[f"{taxa}_n_subgroups"] = len(dataset.data_dict) - 1
    configs["group_level"] = set_group_level
    # Print statistics
    print(f"\n\n\tStatistics of the BIOSCAN-1M Insect dataset with a total of {len(dataset.df.index)}/1,128,308 data samples")
    print("----------------------------------------------------------------------------------------")
    print("\t\t\t\t\t\t\tTaxonomy Group Ranking")
    print("-----------------------------------------------------------------------------------------")
    table = [f"Taxonomy Group Name", "Number of Subgroups", "Number of Not-grouped Samples"]
    print('{:30s} {:25s} {:25s} '.format(table[0], table[1], table[2]))
    print("-----------------------------------------------------------------------------------------\n")
    for cnt, taxa in enumerate(dataset_taxa):
        N1 = group_level_dict[f"{taxa}_n_subgroups"]
        N2 = group_level_dict[f"{taxa}_n_not_grouped_samples"]
        print('G({:1d}): {:18s} {:20d} {:20d} '.format(cnt + 1, taxa, N1, N2))

    DNA_barcode = ["nucraw", "sampleid", "processid", "uri"]
    dataset_barcodes = [bar for bar in DNA_barcode if bar in dataset.df_categories]
    print("\n----------------------------------------------------------------------------------------")
    print("\t\t\t\t\t\t\tBarcode Indexing and Labelling")
    print("-----------------------------------------------------------------------------------------\n")

    cnt = 1
    if "nucraw" in dataset_barcodes:
        print(f"Label ({(cnt)}): Barcode Sequence")
        cnt += 1
    if "uri" in dataset_barcodes:
        print(f"Label ({(cnt)}): Barcode Index Name (BIN)")
        cnt += 1
    if "sampleid" in dataset_barcodes:
        print(f"Label ({(cnt)}): Sample ID Number")
        cnt += 1
    if "processid" in dataset_barcodes:
        print(f"Label ({(cnt)}): BOLD Separation Record Number")
        cnt += 1
    print("\n----------------------------------------End-----------------------------------------------")


def show_statistics(configs, gt_ID, split=''):
    """
    This function shows split statistics.
    :param configs: Configurations.
    :param gt_ID: Ground-truth IDs.
    :param split: Split: all, train, validation, test
    :return:
    """
    if not configs["print_split_statistics"]:
        return

    dataset = BioScan()
    dataset.set_statistics(configs, split=split)

    print_split_statistics(configs, dataset, gt_ID, Set=split)

    plot_split_statistics(dataset.n_sample_per_class, dataset.data_idx_label, len(dataset.data_list),
                          group_level=configs['group_level'], split=split, dataset=configs['exp_name'],
                          fig_path=f"{configs['results_path']}/figures")


def print_split_statistics(configs, dataset, gt_ID, Set=''):
    """
    This function prints Dataset Statistics.
    :param configs: Configurations.
    :param dataset: Dataset class.
    :param gt_ID: Ground-truth IDs.
    :param Set: Split set: all, train, validation, test
    :return:
    """

    label_IDs = {}
    for class_name in dataset.data_dict.keys():
        if class_name not in gt_ID.keys():
            label_IDs[class_name] = 'no_ID'
        else:
            label_IDs[class_name] = gt_ID[class_name]

    table = [f"{configs['group_level']} Name", "Class Number", "Number of Samples"]
    print("\n\n\n------------------------------------------------------------------------")
    print(f"{configs['dataset_name']}\t\tSet:{Set}\t\tGroup Level:{configs['group_level']} \t\t\t\t")
    print("------------------------------------------------------------------------")
    keys = dataset.data_dict.keys()
    data_idx_label = {}
    print('{:27s} {:15s} {:5s} '.format(table[0], table[1], table[2]))
    print("------------------------------------------------------------------------")
    for cnt, key in enumerate(keys):
        data_idx_label[key] = cnt
        if not isinstance(label_IDs[key], str):
            print('{:25s} {:10d} {:20d} '.format(key, label_IDs[key], len(dataset.data_dict[key])))
        else:
            print('{:30s} {:10s} {:15} '.format(key, label_IDs[key], len(dataset.data_dict[key])))
    print("------------------------------------------------------------------------")
    print('{:25s} {:10d} {:22d} '.format("total", cnt + 1, len(dataset.data_list)))
    print("------------------------------------------------------------------------")
    print("no_ID Class(es) are deducted from experiments!")
    print("\n----------------------------------------End-----------------------------------------------")


def plot_split_statistics(sample_num, class_idx, n_samples,
                          group_level='', split='', dataset='', fig_path='', normalize=''):
    """
    This function plots Dataset Statistics.
    :param sample_num: Number of samples per class.
    :param class_idx: Class name vs., numeric IDs.
    :param n_samples: Total number of set samples.
    :param group_level: Taxonomy group level.
    :param split: Split: all, train, validation, test.
    :param dataset: Dataset name.
    :param fig_path: path to save the figures.
    :return:
    """

    class_names = list(class_idx.keys())
    sample_values = sample_num
    if normalize:
        sample_values = [format((num/n_samples), ".2f") for num in sample_num]

    values = np.reshape(sample_values, (1, len(sample_num)))
    if len(class_names) > 20:
        deg_r = 75
    else:
        deg_r = 45

    # Plot the Heatmap with annotations
    plt.figure(figsize=(8, 3))
    ax = sns.heatmap(values, annot=True, cmap='Greens', yticklabels=['Sample Number'],
                     xticklabels=class_names, annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                     cbar=False, fmt='.0f')

    for t in ax.get_xticklabels():
        t.set_rotation(deg_r)
    for t in ax.texts:
        t.set_rotation(90)
    plt.title(f'Class Distribution: {dataset} set: {split}', fontweight='bold', fontsize=12)
    ax.set_xlabel(group_level)
    plt.tight_layout()
    make_directory(fig_path)
    plt.savefig(f"{fig_path}/heatmap_class_{group_level}_{split}_{dataset}.png", dpi=300)
    plt.show()

