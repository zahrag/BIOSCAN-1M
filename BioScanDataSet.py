
import pandas as pd
from torch.utils.data import Dataset


class BioScan(Dataset):

    def __init__(self):
       """
           This class handles getting, setting and showing data statistics ...
           """

    def get_statistics(self, metadata_dir):
       """
           This function sets data attributes read from metadata file of the dataset.
           """

       self.metadata_dir = metadata_dir
       self.df = pd.read_csv(self.metadata_dir, sep='\t')
       self.df = self.get_class_insects(self.df, class_level="Insecta")
       self.processid_list = self.df['processid'].to_list()
       self.phylum_list = self.df['phylum'].to_list()
       self.class_list = self.df['class'].to_list()
       self.order_list = self.df['order'].to_list()
       self.family_list = self.df['family'].to_list()
       self.subfamily_list = self.df['subfamily'].to_list()
       self.genus_list = self.df['genus'].to_list()
       self.species_list = self.df['species'].to_list()
       self.uri_list = self.df['uri'].to_list()
       self.sampleid_list = self.df['sampleid'].to_list()
       self.image_names = self.df['image_file'].to_list()
       self.image_tar = self.df['image_tar'].to_list()
       self.index = self.df.index

    def __len__(self):
        return len(self.img_names)

    def get_class_insects(self, df, class_level="Insecta"):

        """
            This function creates Dataframe of the Insect class only.
            :return: Insect Dataframe
            """

        Insecta_df = [df.iloc[id] for id, cl in enumerate(df['class']) if cl == class_level]

        if len(Insecta_df) == len(df):
            return df

        else:
            print(f"\n{len(df)-len(Insecta_df)} of the samples are NOT Insect class")
            Insecta_df = pd.DataFrame(Insecta_df)
            Insecta_df.reset_index(inplace=True, drop=True)

            return Insecta_df

    def set_statistics(self, data_type="order", metadata_dir=None):
        """

        :param data_type: Type of insect attributes used for processing. There are 7 biological categories defined
        in the dataset, which can be utilized to classify an insect including: "Order", "Phylum", "Class", "Family",
        "Subfamily", "Genus", and "Species".
        In this research we use only "Insect" data in class level.
        In this research we use "Order" information for classifying insects images.

        :param dataset_name: "Original", "Train", "Validation", "Test"
        :return:
        """

        self.get_statistics(metadata_dir)

        # Get data list
        if data_type == "order":
            self.data_list = self.order_list
        elif data_type == "phylum":
            self.data_list = self.phylum_list
        elif data_type == "class":
            self.data_list = self.class_list
        elif data_type == "family":
            self.data_list = self.family_list
        elif data_type == "subfamily":
            self.data_list = self.subfamily_list
        elif data_type == "genus":
            self.data_list = self.genus_list
        elif data_type == "species":
            self.data_list = self.species_list

        # Get the data dictionary
        self.data_dict, self.n_samples_per_class = self.make_data_dict()

        # Get numbered labels of the classes
        self.data_idx_label = self.class_to_ids()

        # Get numbered data samples list
        self.data_list_ids = self.class_to_ids()

    def make_data_dict(self):
        """
        This function create data dict key:label(exe., order name), value:indexes in data list
        :return:
        """

        data_dict = {}
        n_samples_per_class = []
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
        for order in self.order_list:
            data_list_ids.append(self.data_idx_label[order])

        return data_list_ids


def show_statistics(data_type="order", dataset_name="Original", metadata_dir=None, show=False):
    """
         This function shows data statistics from metadata file of the dataset.
         """
    if not show:
        return

    dataset = BioScan()
    dataset.set_statistics(data_type=data_type, metadata_dir=metadata_dir)

    table = [f"{data_type} Name", "label-ID", "Number of Samples"]
    print("\n\n\n--------------------------------------------------------------")
    print(f"\t\t Set:{dataset_name}\t\tType:{data_type} \t\t\t\t")
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

