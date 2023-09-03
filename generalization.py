
import datetime
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torch
from utils import set_seed, load_model, get_model
from tqdm import tqdm
import torch.nn.functional as F
import itertools

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class BioScanTestLoader(Dataset):

    def __init__(self, image_path):
        """
        This function creates dataloader.
        """
        self.image_path = image_path
        self.img_names = os.listdir(image_path)
        self.image_size = 256
        self.crop_size = 224
        self.batch_size = 32
        self.num_workers = 4
        self.transform = transforms.Compose([transforms.Resize(size=self.image_size),
                                            transforms.CenterCrop(size=self.crop_size),
                                            transforms.ToTensor()])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set.
        :param index: index of item in IDs list
        :return: a sample of data as a dict
        """

        image = Image.open(os.path.join(self.image_path, self.img_names[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.img_names[index]


class generalization():

    """ This class predicts Order label of the new images and save the image name with
        its corresponding predicted Order in a txt file """

    def __init__(self, group_level):
        self.group_level = group_level

    def get_keys_by_value(self, dictionary, value):
        return [key for key, val in dictionary.items() if val == value]

    def write_predicted_label(self, image_name, y_pred, gt_labels, date_time, path=""):

        info = ""
        for cnt, image_name in enumerate(image_name):
            taxa_pred = self.get_keys_by_value(gt_labels, y_pred[cnt])
            info += f'{image_name}:{taxa_pred[0]}\n'

        with open(os.path.join(path, f"{self.group_level}_label_predictions_{date_time}.txt"), "w") as fp:
            fp.write(info)
        fp.close()

    def test(self, configs, test_loader):

        set_seed(configs, use_gpu=torch.cuda.is_available())

        best_model = configs['best_model']
        model = get_model(configs, n_classes=len(configs['gt_taxa_labels'].keys()))
        load_model(model, best_model, configs['use_gpu'])
        model.cuda()
        model.eval()
        with torch.no_grad():
            list_test_pred = []
            list_test_proba = []
            list_test_name = []
            for batch_idx, (batch_x_test, batch_x_name) in enumerate(tqdm(test_loader, desc='test', position=0)):
                if configs['use_gpu']:
                    batch_x_test = batch_x_test.cuda()
                batch_output_test = model(batch_x_test)
                batch_proba_test = F.softmax(batch_output_test)
                list_test_pred.append(torch.argmax(batch_output_test, dim=-1))
                list_test_proba.append(batch_proba_test)
                list_test_name.append(batch_x_name)

            y_pred = torch.cat(list_test_proba)
            y_pred_label = torch.cat(list_test_pred)
            y_name = list(itertools.chain(*list_test_name))

        self.write_predicted_label(y_name, y_pred_label,
                                   configs['gt_taxa_labels'],
                                   configs['date_time'], path=os.path.dirname(configs['image_path']))


def get_exp_configs(image_path, model_path):

    """
        This function delivers model configurations ...
        """
    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    config = {
        'date_time': timestamp,
        'image_path': image_path,
        'best_model': os.path.join(model_path, 'large_insect_order_vit_base_patch16_224_CE_s2_weights_best_acc.tar'),
        'group_level': 'order',
        'use_gpu': True,
        'seed': 2,
        'model': 'vit_base_patch16_224',
        'pretrained': True,
        'test': True,
        'gt_taxa_labels': {'Diptera': 0,
                            'Hymenoptera': 1,
                            'Coleoptera': 2,
                            'Hemiptera': 3,
                            'Lepidoptera': 4,
                            'Psocodea': 5,
                            'Thysanoptera': 6,
                            'Trichoptera': 7,
                            'Orthoptera': 8,
                            'Blattodea': 9,
                            'Neuroptera': 10,
                            'Ephemeroptera': 11,
                            'Dermaptera': 12,
                            'Archaeognatha': 13,
                            'Plecoptera': 14,
                            'Embioptera': 15},

    }
    return config


if __name__ == '__main__':

    """
    This script is written to predict Order labels of new Insect images taken by the Centre for Biodiversity Genomics (CBG).
    Additionally the code allows to evaluate generalization capabilities of our best model trained on the cropped and 
    resized BIOSCAN-1M-Insect Large dataset images accessible online through the project's GoogleDrive 
    and zenodo:https://zenodo.org/record/8030065.
    
    To run the code, please assure correct path settings to the directory where the new images (image_path) as well as 
    the model checkpoint (model_path) are saved.
    """

    image_path = ''
    model_path = ''
    configs = get_exp_configs(image_path, model_path)

    test_dataset = BioScanTestLoader(configs['image_path'])
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_dataset.batch_size,
                                 shuffle=False,
                                 num_workers=test_dataset.num_workers)

    generalization = generalization(configs['group_level'])
    generalization.test(configs, test_dataloader)




