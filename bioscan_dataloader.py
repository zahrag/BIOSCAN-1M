from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from BioScanDataSet import BioScan
import io
import h5py
import os


class BioScanLoader(Dataset):

    def __init__(self, configs, data_idx_label, transform=None, split=''):
        """
        This function created dataloader.

        :param configs:
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
        self.metadata_dir = configs[f'metadata_path']

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
            with h5py.File(f'{self.hdf5_dir}/HDF5_BioScan_Part{self.chunk_idx[index]}_CROPPED', 'r') as file:
                dataset = file['bioscan_dataset']
                data = io.BytesIO(np.asarray(dataset[self.img_names[index]]))
                image = Image.open(data)

        elif self.cropped:
            image = Image.open(os.path.join(self.image_dir, "CROPPED_" + self.img_names[index])).convert('RGB')
        else:
            image = Image.open(self.image_dir + self.img_names[index]).convert('RGB')

        return image

    def __getitem__(self, index):
        """
        Generate one item of data set.
        :param index: index of item in IDs list
        :return: a sample of data as a dict
        """

        sample_name = self.sample_list[index]
        label = self.sample_idx_label[sample_name]
        image = self.load_image(index)
        show = False
        if show:
            print('image size:', image.size)
            image.show()

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloader(args, data_idx_label):
    """
    :param args:
    :return: dataloader of train, validation and test sets with data attributes
    """

    if not args['loader']:
        return [], [], [], []

    # ### Train ### #

    if args['no_transform']:
        transform_train = transforms.Compose([transforms.Resize(size=[args['crop_size'], args['crop_size']]),
                                              transforms.ToTensor()])
    else:
        transform_train = transforms.Compose([transforms.Resize(size=args['image_size']),
                                              transforms.RandomCrop(size=args['crop_size']),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

    train_dataset = BioScanLoader(args, data_idx_label, transform=transform_train, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                  num_workers=args['num_workers'])

    # ### Validation ### #
    transform_val = transforms.Compose([transforms.Resize(size=args['image_size']),
                                        transforms.CenterCrop(size=args['crop_size']),
                                        transforms.ToTensor()])

    val_dataset = BioScanLoader(args, data_idx_label, transform=transform_val, split='validation')
    validation_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True,
                                       num_workers=args['num_workers'])

    # ### Test ### #
    test_dataset = BioScanLoader(args, data_idx_label, transform=transform_val, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                 num_workers=args['num_workers'])

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

