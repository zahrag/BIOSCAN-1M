from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from BioScanDataSet import BioScan
import io
import h5py


class BioScanLoader(Dataset):

    def __init__(self, args, transform=None, split=None):

        """
        This function created dataloader.

        :param args:
        :param file_format: The format of dataset.
        :param transform: Transformation
        :param split: "train", "validation", "test"
        """

        self.split = split
        self.data_format = args['data_format']
        self.transform = transform
        self.image_dir = f"{args['dataset_dir']}/{args['dataset_name']}/{args['dataset_name']}_images/"
        self.metadata_dir = f"{args['dataset_dir']}/{args['dataset_name']}/{args['dataset_name']}_{split}_metadata.tsv"
        self.hdf5_dir = f"{args['dataset_dir']}/{args['dataset_name']}/{args['dataset_name']}_hdf5"

        self.dataset = BioScan()
        self.dataset.set_statistics(data_type="order", metadata_dir=self.metadata_dir)

        self.img_names = self.dataset.image_names
        self.order_list = self.dataset.data_list
        self.order_ids = self.dataset.data_idx_label
        self.order_list_ids = self.dataset.data_list_ids
        self.n_samples_per_order = self.dataset.n_samples_per_class
        self.number_of_orders = len(self.order_ids)
        self.number_of_samples = len(self.order_list_ids)
        self.index = self.dataset.index

    def __len__(self):
        return len(self.img_names)

    def load_image(self, index):

        if self.data_format == "hdf5":
            file = h5py.File(self.hdf5_dir, 'r')
            key = list(file.keys())[index]
            data = np.asarray(file[key])
            image = Image.open(io.BytesIO(data))

        else:
            image = Image.open(self.image_dir + self.img_names[index]).convert('RGB')

        return image

    def __getitem__(self, index):
        """
        Generate one item of data set.
        :param index: index of item in IDs list
        :return: a sample of data as a dict
        """

        order_name = self.order_list[index]
        label = self.order_ids[order_name]
        image = self.load_image(index)
        show = False
        if show:
            print('image size:', image.size)
            image.show()

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloader(args):
    """
    :param args:
    :return: dataloader of train, validation and test sets with data attributes
    """

    if not args['loader']:
        return [], [], [], []

    # ### Train ### #
    transform_train = transforms.Compose([transforms.Resize(size=args['image_size']),
                                          transforms.RandomCrop(size=args['crop_size']),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    train_dataset = BioScanLoader(args, transform=transform_train, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                  num_workers=args['num_workers'])

    # ### Validation ### #
    transform_val = transforms.Compose([transforms.Resize(size=args['image_size']),
                                        transforms.CenterCrop(size=args['crop_size']),
                                        transforms.ToTensor()])

    val_dataset = BioScanLoader(args, transform=transform_val, split='validation')
    validation_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True,
                                       num_workers=args['num_workers'])

    # ### Test ### #
    test_dataset = BioScanLoader(args, transform=transform_val, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                 num_workers=args['num_workers'])

    dataset_attributes = {'n_train': train_dataset.number_of_samples,
                          'n_val': val_dataset.number_of_samples,
                          'n_test': test_dataset.number_of_samples,
                          'n_classes': train_dataset.number_of_orders,
                          'class2num_instances': {'train': train_dataset.n_samples_per_order,
                                                  'val': val_dataset.n_samples_per_order,
                                                  'test': test_dataset.n_samples_per_order},
                          'class_to_idx': train_dataset.order_ids}

    return train_dataloader, validation_dataloader, test_dataloader, dataset_attributes

