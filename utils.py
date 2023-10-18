import torch
import torch.nn as nn
import random
import timm
import numpy as np
import tarfile
import zipfile
import os
import shutil
import h5py
import torch.nn.functional as F
import pickle
import io
from PIL import Image
import pandas as pd
import json
import csv
from sklearn.metrics import f1_score

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, inception_v3, mobilenet_v2, densenet121, \
    densenet161, densenet169, densenet201, alexnet, squeezenet1_0, shufflenet_v2_x1_0, wide_resnet50_2, wide_resnet101_2,\
    vgg11, mobilenet_v3_large, mobilenet_v3_small


def set_seed(configs, use_gpu, print_out=True):
    if print_out:
        print('Seed:\t {}'.format(configs['seed']))
    random.seed(configs['seed'])
    np.random.seed(configs['seed'])
    torch.manual_seed(configs['seed'])
    if use_gpu:
        torch.cuda.manual_seed(configs['seed'])


def update_correct_per_class(batch_output, batch_y, d):
    predicted_class = torch.argmax(batch_output, dim=-1)
    for true_label, predicted_label in zip(batch_y, predicted_class):
        if true_label == predicted_label:
            d[true_label.item()] += 1
        else:
            d[true_label.item()] += 0


def update_correct_per_class_topk(batch_output, batch_y, d, k):
    topk_labels_pred = torch.argsort(batch_output, axis=-1, descending=True)[:, :k]
    for true_label, predicted_labels in zip(batch_y, topk_labels_pred):
        d[true_label.item()] += torch.sum(true_label == predicted_labels).item()


def update_correct_per_class_avgk(val_probas, val_labels, d, lmbda):
    ground_truth_probas = torch.gather(val_probas, dim=1, index=val_labels.unsqueeze(-1))
    for true_label, predicted_label in zip(val_labels, ground_truth_probas):
        d[true_label.item()] += (predicted_label >= lmbda).item()


def count_correct_topk(scores, labels, k):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    top_k_scores = torch.argsort(scores, axis=-1, descending=True)[:, :k]
    labels = labels.view(len(labels), 1)
    return torch.eq(labels, top_k_scores).sum()


def count_correct_avgk(probas, labels, lmbda):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    gt_probas = torch.gather(probas, dim=1, index=labels.unsqueeze(-1))
    res = torch.sum((gt_probas) >= lmbda)
    return res


def load_model(model, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    model.load_state_dict(d['model'])
    return d['epoch']


def load_optimizer(optimizer, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    optimizer.load_state_dict(d['optimizer'])


def save(model, optimizer, epoch, location):
    dir = os.path.dirname(location)
    make_directory(dir)

    d = {'epoch': epoch,
         'model': model.state_dict(),
         'optimizer': optimizer.state_dict()}
    torch.save(d, location)


def decay_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    print('Switching lr to {}'.format(optimizer.param_groups[0]['lr']))
    return optimizer


def update_optimizer(optimizer, lr_schedule, epoch):
    if epoch in lr_schedule:
        optimizer = decay_lr(optimizer)
    return optimizer


def get_model(configs, n_classes):
    pytorch_models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
                      'resnet152': resnet152, 'densenet121': densenet121, 'densenet161': densenet161,
                      'densenet169': densenet169, 'densenet201': densenet201, 'mobilenet_v2': mobilenet_v2,
                      'inception_v3': inception_v3, 'alexnet': alexnet, 'squeezenet': squeezenet1_0,
                      'shufflenet': shufflenet_v2_x1_0, 'wide_resnet50_2': wide_resnet50_2,
                      'wide_resnet101_2': wide_resnet101_2, 'vgg11': vgg11, 'mobilenet_v3_large': mobilenet_v3_large,
                      'mobilenet_v3_small': mobilenet_v3_small
                      }
    timm_models = {'inception_resnet_v2', 'inception_v4', 'efficientnet_b0', 'efficientnet_b1',
                   'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'vit_base_patch16_224',
                   'vit_small_patch16_224'
                   }

    if configs['model'] in pytorch_models.keys() and not configs['pretrained']:
        if configs['model'] == 'inception_v3':
            model = pytorch_models[configs['model']](pretrained=False, num_classes=n_classes, aux_logits=False)
        else:
            model = pytorch_models[configs['model']](pretrained=False, num_classes=n_classes)
    elif configs['model'] in pytorch_models.keys() and configs['pretrained']:
        if configs['model'] in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2',
                          'wide_resnet101_2', 'shufflenet'}:
            model = pytorch_models[configs['model']](pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_classes)
        elif configs['model'] in {'alexnet', 'vgg11'}:
            model = pytorch_models[configs['model']](pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_classes)
        elif configs['model'] in {'densenet121', 'densenet161', 'densenet169', 'densenet201'}:
            model = pytorch_models[configs['model']](pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_classes)
        elif configs['model'] == 'mobilenet_v2':
            model = pytorch_models[configs['model']](pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        elif configs['model'] == 'inception_v3':
            model = inception_v3(pretrained=True, aux_logits=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_classes)
        elif configs['model'] == 'squeezenet':
            model = pytorch_models[configs['model']](pretrained=True)
            model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = n_classes
        elif configs['model'] == 'mobilenet_v3_large' or configs['model'] == 'mobilenet_v3_small':
            model = pytorch_models[configs['model']](pretrained=True)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, n_classes)

    elif configs['model'] in timm_models:
        model = timm.create_model(configs['model'], pretrained=configs['pretrained'], num_classes=n_classes)
    else:
        raise NotImplementedError

    return model


def open_pickle(path):

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    results = objects[0]

    return results


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_directory(path):
    shutil.rmtree(path, ignore_errors=False, onerror=None)


def remove_file(file, path):
    os.remove(os.path.join(path, file))


def move_to_dir(source=None, destination=None):
    make_directory(destination)
    shutil.move(source, destination)


def copy_to_dir(source=None, destination=None):
    make_directory(destination)
    shutil.copy(source, destination)


def create_zip(source_folder=None, output_zip=None):

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join(f"bioscan/images/{os.path.splitext(os.path.basename(output_zip))[0]}", file)
                zipf.write(file_path, arcname=arcname)


def extract_zip(zip_file=None, path=None):

    make_directory(path)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(path)


def create_tar(name=None, path=None):
    make_directory(path)
    print(path)
    with tarfile.open(name, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))


def extract_tar(tar_file=None, path=None):
    make_directory(path)
    tar_file = tarfile.open(tar_file)
    tar_file.extractall(path)
    tar_file.close()


def make_tsv(df, name=None, path=None):
    make_directory(path)
    df.to_csv(os.path.join(path, name), sep='\t', index=False)


def read_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
    return df


def convert_tsv_to_jsonld(tsv_file, jsonld_file):

    # Read data from a TSV file
    data = []
    with open(tsv_file, 'r', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)

    # Write data in a JSON-LD file
    with open(jsonld_file, 'w') as file:
        json.dump(data, file, indent=4)


def read_jsonld(jsonld_file):
    with open(jsonld_file, 'r') as file:
        data = json.load(file)
    return data


def resize_image(input_file, output_file, resize_dimension=256):
    make_directory(os.path.dirname(output_file))
    command = f'convert "{input_file}" -resize x{resize_dimension} "{output_file}"'
    os.system(command)


def create_hdf5(date_time, dataset_name='', path='', data_typ='Original Full Size', author='Zahra Gharaee'):

    with h5py.File(path, 'w') as hdf5:
        dataset = hdf5.create_group(dataset_name)
        dataset.attrs['Description'] = f'BIOSCAN_1M Insect Dataset: {data_typ} Images'
        dataset.attrs['Copyright Holder'] = 'CBG Photography Group'
        dataset.attrs['Copyright Institution'] = 'Centre for Biodiversity Genomics (email:CBGImaging@gmail.com)'
        dataset.attrs['Photographer'] = 'CBG Robotic Imager'
        dataset.attrs['Copyright License'] = 'Creative Commons-Attribution Non-Commercial Share-Alike (CC BY-NC-SA 4.0)'
        dataset.attrs['Copyright Contact'] = 'collectionsBIO@gmail.com'
        dataset.attrs['Copyright Year'] = '2021'
        dataset.attrs['Author'] = author
        dataset.attrs['Date'] = date_time

    return dataset


def write_in_hdf5(hdf5, image, image_name, image_dir=None, save_binary=False):
    """
    This function writes an image in a HDF5 file.
    :param hdf5: HDF5 file to write image in.
    :param image: Image as data array.
    :param image_name: Name which image is archived with.
    :param image_dir: Directory where the image is saved.
    :param save_binary: If save as binary data (compressed) to save space?
    :return:
    """

    if save_binary:
        if image_dir is not None:
            with open(image_dir, 'rb') as img_f:
                binary_data = img_f.read()
            image_data = np.asarray(binary_data)
        else:
            binary_data_io = io.BytesIO()
            image.save(binary_data_io, format='JPEG')
            binary_data = binary_data_io.getvalue()
            image_data = np.frombuffer(binary_data, dtype=np.uint8)
    else:
        image_data = np.array(image)

    hdf5.create_dataset(image_name, data=image_data)


def read_from_hdf5(hdf5, image_name, saved_as_binary=False):
    """
    This function reads an image from a HDF5 file.
    :param hdf5: The Hdf5 file to read from.
    :param image_name: The image to read.
    :param saved_as_binary: If data is saved as binary?
    :return:
    """

    if saved_as_binary:
        data = np.array(hdf5[image_name])
        image = Image.open(io.BytesIO(data))
    else:
        data = np.asarray(hdf5[image_name])
        image = Image.fromarray(data)

    return image


def get_f1_score(y_true, y_pred, metric='micro'):
    """ This function computes F1-Score using metrics: 'micro', 'macro', 'weighted', None """
    return f1_score(y_true.cpu().numpy().tolist(), torch.argmax(y_pred, dim=-1).cpu().numpy().tolist(), average=metric)


class MulticlassFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(MulticlassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss