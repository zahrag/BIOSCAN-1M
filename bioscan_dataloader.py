from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from BioScanDataSet import BioScan
import io
import h5py
import sys
import os

class BioScanLoader(Dataset):

    def __init__(self, configs, data_idx_label, transform=None, split=''):
        """
        This function creates dataloader.
        :param configs: Configurations
        :param data_idx_label: Ground-Truth Class Label-IDs
        :param transform: Transformation
        :param split: "train", "validation", "test"
        """

        self.split = split
        self.transform = transform
        self.cropped = configs['cropped']
        self.data_format = configs['data_format']
        self.image_dir = configs['image_path']
        self.hdf5_dir = configs['hdf5_path']

        self.dataset = BioScan()
        self.dataset.set_statistics(configs, split=split)

        self.sample_idx_label = data_idx_label
        self.img_names = self.dataset.image_names
        self.sample_list = self.dataset.data_list
        self.n_samples_per_class = self.dataset.get_n_sample_class(self.dataset.data_dict)
        self.number_of_class = len(data_idx_label)
        self.number_of_samples = len(self.sample_list)
        self.chunk_idx = self.dataset.chunk_index
        self.chunk_length = self.dataset.chunk_length

    def __len__(self):
        return len(self.img_names)

    def load_image(self, index):

        if self.data_format == "hdf5":
            with h5py.File(self.hdf5_dir, 'r') as file:
                dataset = file['bioscan_dataset']
                data = io.BytesIO(np.asarray(dataset[self.img_names[index]]))
                image = Image.open(data)

        elif self.data_format == "folder":

            # If using BIOSCAN_1M_Insect dataset structure with images in 113 folders (part1:part113)
            img_dir = os.path.join(self.image_dir, f"part{self.chunk_idx[index]}")
            # If all images in one folder
            # img_dir = self.image_dir
            image = Image.open(img_dir + self.img_names[index]).convert('RGB')
        else:
            sys.exit("Wrong data_format: " + self.data_format + " does not exist.")

        return image

    def __getitem__(self, index):
        """
        Generate one item of data set.
        :param index: index of item in IDs list
        :return: a sample of data as a dict
        """

        class_name = self.sample_list[index]
        label = self.sample_idx_label[class_name]
        image = self.load_image(index)
        show = False
        if show:
            print('image size:', image.size)
            image.show()

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloader(configs, data_idx_label):
    """
    :param configs:
    :return: dataloader of train, validation and test sets with data attributes
    """

    if not configs['loader']:
        return [], [], [], []

    # ### Train ### #

    if configs['no_transform']:
        transform_train = transforms.Compose([transforms.Resize(size=[configs['crop_size'], configs['crop_size']]),
                                              transforms.ToTensor()])
    else:
        transform_train = transforms.Compose([transforms.Resize(size=configs['image_size']),
                                              transforms.RandomCrop(size=configs['crop_size']),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

    train_dataset = BioScanLoader(configs, data_idx_label, transform=transform_train, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True,
                                  num_workers=configs['num_workers'])

    # ### Validation ### #
    transform_val = transforms.Compose([transforms.Resize(size=configs['image_size']),
                                        transforms.CenterCrop(size=configs['crop_size']),
                                        transforms.ToTensor()])

    val_dataset = BioScanLoader(configs, data_idx_label, transform=transform_val, split='validation')
    validation_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=True,
                                       num_workers=configs['num_workers'])

    # ### Test ### #
    test_dataset = BioScanLoader(configs, data_idx_label, transform=transform_val, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False,
                                 num_workers=configs['num_workers'])

    dataset_attributes = {'n_train': train_dataset.number_of_samples,
                          'n_val': val_dataset.number_of_samples,
                          'n_test': test_dataset.number_of_samples,
                          'n_classes': train_dataset.number_of_class,
                          'class2num_instances': {'train': list(train_dataset.n_samples_per_class),
                                                  'val': list(val_dataset.n_samples_per_class),
                                                  'test': list(test_dataset.n_samples_per_class),
                                                  },
                          'class_to_idx': data_idx_label}

    return train_dataloader, validation_dataloader, test_dataloader, dataset_attributes

