
import pandas as pd
from torch.utils.data import Dataset
import math


class BioScan(Dataset):
    def __init__(self):
       """
           This class handles getting, setting and showing data statistics ...
           """

    def get_statistics(self, metadata_dir):
        """
           This function sets data attributes read from metadata file of the dataset.
           This includes biological taxonomy information, DNA barcode indexes and RGB image labels.
           """

        self.metadata_dir = metadata_dir
        self.df = pd.read_csv(self.metadata_dir, sep='\t')
        self.df = self.get_class_insects(self.df, class_level="Insecta", check=False)
        self.index = self.df.index.to_list()
        self.df_categories = self.df.keys().to_list()

        # Biological Taxonomy
        if 'phylum' in self.df_categories:
            self.phylum_list = self.df['phylum'].to_list()
        if 'class' in self.df_categories:
            self.class_list = self.df['class'].to_list()
        if 'order' in self.df_categories:
            self.order_list = self.df['order'].to_list()
        if 'family' in self.df_categories:
            self.family_list = self.df['family'].to_list()
        if 'subfamily' in self.df_categories:
            self.subfamily_list = self.df['subfamily'].to_list()
        if 'genus' in self.df_categories:
            self.genus_list = self.df['genus'].to_list()
        if 'species' in self.df_categories:
            self.species_list = self.df['species'].to_list()
        if 'subspecies' in self.df_categories:
           self.subspecies_list = self.df['subspecies'].to_list()
        if 'tribe' in self.df_categories:
           self.tribes_list = self.df['tribe'].to_list()
        if 'name' in self.df_categories:
            self.names_list = self.df['name'].to_list()

        # Barcode and data Indexing
        if 'nucraw' in self.df_categories:
           self.barcods_list = self.df['nucraw'].to_list()
        if 'processid' in self.df_categories:
            self.processid_list = self.df['processid'].to_list()
        if 'sampleid' in self.df_categories:
            self.sampleid_list = self.df['sampleid'].to_list()
        if 'uri' in self.df_categories:
            self.uri_list = self.df['uri'].to_list()

        # Image Indexes
        if 'image_tar' in self.df_categories:
           self.image_tar = self.df['image_tar'].to_list()
        if 'image_file' in self.df_categories:
           self.image_names = self.df['image_file'].to_list()
        else:
           self.image_names = self.df['sampleid'].to_list()

    def __len__(self):
        return len(self.index)

    def get_class_insects(self, df, class_level="Insecta", check=False):
        """
            This function creates Dataframe of the Insect class only.
            :return: Insect Dataframe
            """

        if not check:
            return df

        Insecta_df = [df.iloc[id] for id, cl in enumerate(df['class']) if cl == class_level]
        if len(Insecta_df) == len(df):
            return df

        else:
            print(f"\n{len(df)-len(Insecta_df)} of the samples are NOT Insect class")
            Insecta_df = pd.DataFrame(Insecta_df)
            Insecta_df.reset_index(inplace=True, drop=True)

            return Insecta_df

    def set_statistics(self, group_level="order", metadata_dir=None):
        """
        :param group_level: Group level of insect attributes used for processing. There are 9 biological categories defined
        in the dataset, which can be utilized to classify an insect including: "Order", "Phylum", "Class", "Family",
        "Subfamily", "Genus", "Tribe", "Species" and "SubSpecies".
        In this research we use only "Insect" data in class level.
        In this research we use "Order" information for classifying insects images.

        :param dataset_name: "Small", "Medium", "Large", "Train", "Validation", "Test"
        :return:
        """

        self.get_statistics(metadata_dir)

        # Get data list as one of the Biological Taxonomy
        if group_level == "order":
            self.data_list = self.order_list
        elif group_level == "phylum":
            self.data_list = self.phylum_list
        elif group_level == "class":
            self.data_list = self.class_list
        elif group_level == "family":
            self.data_list = self.family_list
        elif group_level == "subfamily":
            self.data_list = self.subfamily_list
        elif group_level == "genus":
            self.data_list = self.genus_list
        elif group_level == "species":
            self.data_list = self.species_list
        elif group_level == "subspecies":
            self.data_list = self.subspecies_list
        elif group_level == "tribe":
            self.data_list = self.tribes_list
        elif group_level == "name":
            self.data_list = self.names_list

        # Get the data dictionary
        self.data_dict, self.n_samples_per_class = self.make_data_dict()

        # Get numbered labels of the classes
        self.data_idx_label = self.class_to_ids()

        # Get numbered data samples list
        self.data_list_ids = self.class_list_idx()

    def make_data_dict(self):
        """
        This function create data dict key:label(exe., order name), value:indexes in data list
        :return:
        """

        data_dict = {}
        n_samples_per_class = []
        for cnt, data in enumerate(self.data_list):
            if not isinstance(data, str):
                if math.isnan(data):
                    self.data_list[cnt] = "not_classified"
                else:
                    print("Not a string type data name is detected!")

        data_names = []
        for data in self.data_list:
            if data not in data_names:
                data_names.append(data)

        for name in data_names:
            indexes = [ind for ind in self.index if self.data_list[ind] == name]
            data_dict[name] = indexes
            n_samples_per_class.append(len(indexes))

        return data_dict, n_samples_per_class

    def class_to_ids(self):
        """
        This function create index labels (order to numbered labels).
        :return:
        """
        data_idx_label = {}
        for num_id, key in enumerate(self.data_dict.keys()):
            data_idx_label[key] = num_id

        return data_idx_label

    def class_list_idx(self):
        """
        This function create data list of numbered labels.
        :return:
        """

        data_list_ids = []
        for order in self.data_list:
            data_list_ids.append(self.data_idx_label[order])

        return data_list_ids


def show_dataset_statistics(dataset_name="large_dataset", metadata_dir=None, show=False):
    """
         This function shows data statistics from metadata file of the dataset.
         """
    if not show:
        return

    print("\n\nCreating data statistics ...")
    print("Attention:This process can take time especially if the dataset is big!")

    dataset = BioScan()
    dataset.get_statistics(metadata_dir=metadata_dir)

    # Get taxonomy ranking statistics
    taxa_gt_sored = ["domain", "kingdom", "phylum", "class", "order", "family", "subfamily", "tribe", "genus",
                     "species", "subspecies"]
    dataset_taxa = [taxa for taxa in taxa_gt_sored if taxa in dataset.df_categories]

    # Get subgroups statistics
    group_level_dict = {}
    for taxa in dataset_taxa:
        dataset.set_statistics(group_level=taxa, metadata_dir=metadata_dir)
        group_level_dict[f"{taxa}_n_subgroups"] = len(dataset.data_dict)
        group_level_dict[f"{taxa}_n_not_grouped_samples"] = 0
        if "not_classified" in dataset.data_dict:
            group_level_dict[f"{taxa}_n_not_grouped_samples"] = len(dataset.data_dict["not_classified"])
            group_level_dict[f"{taxa}_n_subgroups"] = len(dataset.data_dict)-1

    # Show statistics
    print(f"\n\n\t\t\t\tDataset: {dataset_name} with a total of {len(dataset.df.index)} data samples")
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
    print("\n----------------------------------------End-----------------------------------------------")


def show_statistics(group_level="order", dataset_name="large_dataset", metadata_dir=None, show=False):

    """
     This function shows data statistics from metadata file of the dataset.
     """
    if not show:
        return

    dataset = BioScan()
    dataset.set_statistics(group_level=group_level, metadata_dir=metadata_dir)

    table = [f"{group_level} Name", "label-ID", "Number of Samples"]
    print("\n\n\n--------------------------------------------------------------")
    print(f"\t\t Set:{dataset_name}\t\tType:{group_level} \t\t\t\t")
    print("--------------------------------------------------------------")
    keys = dataset.data_dict.keys()
    data_idx_label = {}
    print('{:25s} {:15s} {:5s} '.format(table[0], table[1], table[2]))
    print("--------------------------------------------------------------")
    for cnt, key in enumerate(keys):
        data_idx_label[key] = cnt
        print('{:18s} {:10d} {:20d} '.format(key, cnt + 1, len(dataset.data_dict[key])))
    print("--------------------------------------------------------------")
    print('{:18s} {:10d} {:25d} '.format("total", cnt + 1, len(dataset.data_list)))
    print("--------------------------------------------------------------")

