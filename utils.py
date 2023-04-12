import torch
import torch.nn as nn
import random
import timm
import numpy as np
import tarfile
import os
import shutil
import pandas as pd
from tqdm import tqdm
import h5py


from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, inception_v3, mobilenet_v2, densenet121, \
    densenet161, densenet169, densenet201, alexnet, squeezenet1_0, shufflenet_v2_x1_0, wide_resnet50_2, wide_resnet101_2,\
    vgg11, mobilenet_v3_large, mobilenet_v3_small


def set_seed(args, use_gpu, print_out=True):
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed(args.seed)


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


def get_model(args, n_classes):
    pytorch_models = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
                      'resnet152': resnet152, 'densenet121': densenet121, 'densenet161': densenet161,
                      'densenet169': densenet169, 'densenet201': densenet201, 'mobilenet_v2': mobilenet_v2,
                      'inception_v3': inception_v3, 'alexnet': alexnet, 'squeezenet': squeezenet1_0,
                      'shufflenet': shufflenet_v2_x1_0, 'wide_resnet50_2': wide_resnet50_2,
                      'wide_resnet101_2': wide_resnet101_2, 'vgg11': vgg11, 'mobilenet_v3_large': mobilenet_v3_large,
                      'mobilenet_v3_small': mobilenet_v3_small
                      }
    timm_models = {'inception_resnet_v2', 'inception_v4', 'efficientnet_b0', 'efficientnet_b1',
                   'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'vit_base_patch16_224'}

    if args.model in pytorch_models.keys() and not args.pretrained:
        if args.model == 'inception_v3':
            model = pytorch_models[args.model](pretrained=False, num_classes=n_classes, aux_logits=False)
        else:
            model = pytorch_models[args.model](pretrained=False, num_classes=n_classes)
    elif args.model in pytorch_models.keys() and args.pretrained:
        if args.model in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2',
                          'wide_resnet101_2', 'shufflenet'}:
            model = pytorch_models[args.model](pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_classes)
        elif args.model in {'alexnet', 'vgg11'}:
            model = pytorch_models[args.model](pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_classes)
        elif args.model in {'densenet121', 'densenet161', 'densenet169', 'densenet201'}:
            model = pytorch_models[args.model](pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_classes)
        elif args.model == 'mobilenet_v2':
            model = pytorch_models[args.model](pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, n_classes)
        elif args.model == 'inception_v3':
            model = inception_v3(pretrained=True, aux_logits=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_classes)
        elif args.model == 'squeezenet':
            model = pytorch_models[args.model](pretrained=True)
            model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = n_classes
        elif args.model == 'mobilenet_v3_large' or args.model == 'mobilenet_v3_small':
            model = pytorch_models[args.model](pretrained=True)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, n_classes)

    elif args.model in timm_models:
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=n_classes)
    else:
        raise NotImplementedError

    return model


def make_directory(path):

    if not os.path.exists(path):
        os.makedirs(path)


def extract_tar(tar_file=None, path=None):

    make_directory(path)
    tar_file = tarfile.open(tar_file)
    tar_file.extractall(path)  # specify which folder to extract to
    tar_file.close()


def move_to_dir(source=None, destination=None):

    make_directory(destination)
    shutil.move(source, destination)


def copy_to_dir(source=None, destination=None):

    make_directory(destination)
    shutil.copy(source, destination)


def make_tar(name=None, path=None):

    make_directory(path)
    print(path)
    with tarfile.open(name, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))


def make_tsv(file, name=None, path=None):

    make_directory(path)
    file.to_csv(path + name, sep='\t', index=False)


def save_configs(datetime, config, log_dir=None):

    info = f"Configurations of Experiment Run on {datetime}\n"
    for item in config.keys():
        info += f'{item}:{config[item]}\n'

    with open(log_dir + f"{config['exp_name']}_configs.txt", "w") as fp:
        fp.write(info)
    fp.close()


def make_path_configs(config, timestamp):

    if config["train"]:
        save_dir = os.path.join(os.getcwd(), 'results')
        save_dir += "/{timestamp:s}_{dataset:s}_epoch{epoch:d}/".format(timestamp=timestamp,
                                                                        dataset=config['dataset_name'],
                                                                        epoch=config["n_epochs"])
        make_directory(save_dir)
        save_configs(timestamp, config, log_dir=save_dir)
        config["results_dir"] = save_dir
        config["exp_name"] = config["dataset_name"]

    return config


def make_hdf5(dataset_name='small_dataset', path=None):

    dataset_dir = f"{path}/{dataset_name}"
    hdf5_file = h5py.File(f"{dataset_dir}/{dataset_name}_hdf5", 'w')
    df = pd.read_table(f"{dataset_dir}/{dataset_name}_metadata.tsv")
    list_of_dict = df.to_dict(orient='records')
    for image_data in tqdm(list_of_dict):
        orgpic_id = image_data['orgpic_id']
        image_dir = f"{dataset_dir}/{dataset_name}_images/{image_data['image_file']}"
        with open(image_dir, 'rb') as img_f:
            binary_data = img_f.read()
        binary_data_np = np.asarray(binary_data)
        hdf5_file.create_dataset(str(orgpic_id), data=binary_data_np)
