
import pandas as pd
from torch.utils.data import Dataset
import math
import os


class BioScan(Dataset):
    def __init__(self):
       """
           This class handles getting, setting and showing data statistics ...
           """

    def get_statistics(self, metadata_dir, split='', exp=''):
        """
           This function sets data attributes read from metadata file of the dataset.
           This includes biological taxonomy information, DNA barcode indexes and RGB image labels.
           """

        # Get Samples Roles In Experiments
        self.experiment_names = ['insect_order_level', 'diptera_family_level',                # Large Dataset
                                 'medium_insect_order_level', 'medium_diptera_family_level',  # Medium Dataset
                                 'small_insect_order_level', 'small_diptera_family_level',    # Small Dataset
                                 ]

        self.metadata_dir = metadata_dir
        self.df = self.read_metadata(metadata_dir, split, exp)
        self.index = self.df.index.to_list()
        self.df_categories = self.df.keys().to_list()
        self.n_DatasetAttributes = len(self.df_categories)

        # Biological Taxonomy
        self.taxa_gt_sorted = {'0': 'domain',      '1': 'kingdom',   '2': 'phylum', '3': 'class', '4': 'order',
                               '5': 'family',      '6': 'subfamily', '7': 'tribe',  '8': 'genus', '9': 'species',
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

    def read_metadata(self, metadata_dir, split, exp):

        if os.path.isfile(metadata_dir):
            df = pd.read_csv(metadata_dir, sep='\t', low_memory=False)
        else:
            raise RuntimeError(f"Not a metadata to read in:\n{metadata_dir}")

        if not split:
            if exp in self.experiment_names[:2]:
                return df
            elif exp in self.experiment_names[2:]:
                df_set = [df.iloc[id] for id, cl in enumerate(df[exp]) if cl in ['train', 'validation', 'test']]
                set = ['Medium' if ''.join(list(exp)[:3]) == 'med' else 'Small']
                # print(f"\n{len(df_set)} of the samples are {set[0]} set for experiment {exp}.")
                df_set = pd.DataFrame(df_set)
                df_set.reset_index(inplace=True, drop=True)
                return df_set
            else:
                raise RuntimeError(f"No an experiment is set!")

        # Get the split metadata
        if exp not in df.columns:
            raise RuntimeError("Split for the experiments is NOT available: Do split first!")

        df_split = [df.iloc[id] for id, cl in enumerate(df[exp]) if cl == split]
        # print(f"\n{len(df_split)} of the samples are {split} of {exp}.")
        df_split = pd.DataFrame(df_split)
        df_split.reset_index(inplace=True, drop=True)

        return df_split

    def set_statistics(self, configs, split=''):
        """

        :param configs: Arguments
        :return:
        """

        self.get_statistics(configs["metadata_path"], exp=configs["exp_name"], split=split)

        # Get data list as one of the Biological Taxonomy
        if configs["group_level"] in self.taxonomy_groups_list_dict.keys():
            self.data_list = self.taxonomy_groups_list_dict[configs["group_level"]]

        else:
            print(f'Dataset Does NOT contain Taxonomy Group Ranking {configs["group_level"]}')

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
        This function create data dict key:label(exe., order name), value:indexes in data list
        :return:
        """

        data_dict = {}
        for cnt, data in enumerate(data_list):
            if not isinstance(data, str):
                if math.isnan(data):
                    data_list[cnt] = "not_classified"
                else:
                    print("Not a string type data name is detected!")

        data_names = []
        for data in data_list:
            if data not in data_names:
                data_names.append(data)

        for name in data_names:
            indexes = [ind for ind in index if data_list[ind] == name]
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
        This function create index labels (order to numbered labels).
        :return:
        """
        data_idx_label = {}
        data_names = list(data_dict.keys())
        for name in data_names:
            data_idx_label[name] = data_names.index(name)

        return data_idx_label

    def get_n_sample_class(self, data_dict):

        data_samples_list = list(data_dict.values())
        n_sample_per_class = [len(class_samples) for class_samples in data_samples_list]

        return n_sample_per_class

    def class_list_idx(self, data_list, data_idx_label):
        """
        This function create data list of numbered labels.
        :return:
        """

        data_list_ids = []
        for data in data_list:
            data_list_ids.append(data_idx_label[data])

        return data_list_ids


def show_dataset_statistics(configs):
    """
         This function shows data statistics from metadata file of the dataset.
         """
    if not configs["print_statistics"]:
        return

    print("\n\nCreating data statistics ...")
    print("ATTENTION:This process can take time especially if the dataset is big!")

    dataset = BioScan()
    dataset.get_statistics(metadata_dir=configs["metadata_path"], exp=configs["exp_name"])

    print("\n\n----------------------------------------------------------------------------------------")
    print(f"\t\t\t\t\t\t\t\tCopyright")
    print("----------------------------------------------------------------------------------------")
    print("Copyright Holder: CBG Photography Group")
    print("Copyright Institution: Centre for Biodiversity Genomics (email:CBGImaging@gmail.com)")
    print("Photographer: CBG Robotic Imager")
    print("Copyright: Creative Commons-Attribution Non-Commercial Share-Alike")
    print("Copyright Contact: collectionsBIO@gmail.com")
    print("----------------------------------------------------------------------------------------")

    # Get taxonomy ranking statistics
    dataset_taxa = [taxa for taxa in dataset.taxa_gt_sorted.values() if taxa in dataset.df_categories]

    set_group_level = configs["group_level"]
    # Get subgroups statistics
    group_level_dict = {}
    for taxa in dataset_taxa:
        configs["group_level"] = taxa
        dataset.set_statistics(configs)
        group_level_dict[f"{taxa}_n_subgroups"] = len(dataset.data_dict)
        group_level_dict[f"{taxa}_n_not_grouped_samples"] = 0
        if "not_classified" in dataset.data_dict:
            group_level_dict[f"{taxa}_n_not_grouped_samples"] = len(dataset.data_dict["not_classified"])
            group_level_dict[f"{taxa}_n_subgroups"] = len(dataset.data_dict)-1
    configs["group_level"] = set_group_level

    # Show statistics
    print(f"\n\n\t\t\tStatistics of the {configs['dataset_name']} with a total of {len(dataset.df.index)} data samples")
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


def show_statistics(configs, gt_ID='', split=''):
    """
         This function shows data statistics from metadata file of the dataset.
         """
    if not configs["print_split_statistics"]:
        return

    dataset = BioScan()
    dataset.set_statistics(configs, split=split)

    Set = split
    if split not in ['train', 'validation', 'test']:
        Set = 'All'

    label_IDs = {}
    for class_name in dataset.data_dict.keys():
        if class_name not in gt_ID.keys():
            label_IDs[class_name] = 'no_ID'
        else:
            label_IDs[class_name] = gt_ID[class_name]

    table = [f"{configs['group_level']} Name", "Class Number", "Number of Samples"]
    print("\n\n\n------------------------------------------------------------------------")
    print(f"{configs['dataset_name']}\t\tSet:{Set}\tGroup Level:{configs['group_level']} \t\t\t\t")
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
    print("------------------------------------------------------------------------")

