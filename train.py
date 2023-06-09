import os
from tqdm import tqdm
import pickle
import time
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from utils import set_seed, save, get_model, update_optimizer
from epoch import train_epoch, val_epoch
from torch.utils.tensorboard import SummaryWriter
from utils import MulticlassFocalLoss

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(configs, train_loader, val_loader, dataset_attributes):

    if not configs['train']:
        return

    set_seed(configs, use_gpu=torch.cuda.is_available())
    save_dir = configs['results_path']
    save_name = configs['exp_name']

    writer = SummaryWriter(configs['log'] + "/")

    model = get_model(configs, n_classes=dataset_attributes['n_classes'])

    criteria = CrossEntropyLoss()

    if configs['loss'] == "Focal":
        criteria = MulticlassFocalLoss(gamma=2)

    if configs['use_gpu']:
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        criteria.cuda()

    optimizer = SGD(model.parameters(),
                    lr=configs['lr'], momentum=configs['momentum'], weight_decay=configs['mu'], nesterov=True)

    # Containers for storing metrics over epochs
    loss_train, acc_train, topk_acc_train = [], [], []
    loss_val, acc_val, topk_acc_val, avgk_acc_val, class_acc_val, macro_topk_acc_val = [], [], [], [], [], []

    print('args.k : ', configs['k'])

    lmbda_best_acc = None
    best_val_acc = float('-inf')

    for epoch in tqdm(range(configs['n_epochs']), desc='epoch', position=0):

        t = time.time()
        optimizer = update_optimizer(optimizer, lr_schedule=configs['epoch_decay'], epoch=epoch)

        # #### Training
        loss_epoch_train, acc_epoch_train, topk_acc_epoch_train = train_epoch(model, optimizer, train_loader,
                                                                              criteria, loss_train, acc_train,
                                                                              topk_acc_train, configs['k'],
                                                                              dataset_attributes['n_train'],
                                                                              configs['use_gpu'])

        writer.add_scalar('loss_epoch_train', loss_epoch_train, epoch)
        writer.add_scalar('acc_epoch_train', acc_epoch_train, epoch)
        for k in configs['k']:
            writer.add_scalar(f'topk_acc_epoch_train-Top{k}:', topk_acc_epoch_train[k], epoch)

        # #### Validation
        loss_epoch_val, acc_epoch_val, topk_acc_epoch_val, \
        avgk_acc_epoch_val, macro_topk_acc_epoch_val, lmbda_val = val_epoch(model, val_loader, criteria,
                                                                            loss_val, acc_val, topk_acc_val, avgk_acc_val,
                                                                            class_acc_val, macro_topk_acc_val,
                                                                            configs['k'], dataset_attributes,
                                                                            configs['use_gpu'])

        writer.add_scalar('loss_epoch_val', loss_epoch_val, epoch)
        writer.add_scalar('acc_epoch_val', acc_epoch_val, epoch)
        for k in configs['k']:
            writer.add_scalar(f'avgk_acc_epoch_val-Top{k}:', avgk_acc_epoch_val[k], epoch)
            writer.add_scalar(f'lmbda_val-Top{k}:', lmbda_val[k], epoch)
            writer.add_scalar(f'topk_acc_epoch_val-Top{k}:', topk_acc_epoch_val[k], epoch)

        # Save model at every epoch
        save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights.tar'))

        # save model with best val accuracy
        if acc_epoch_val > best_val_acc:
            best_val_acc = acc_epoch_val
            lmbda_best_acc = lmbda_val
            save(model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights_best_acc.tar'))

        # Print statistics
        print()
        print(f'epoch {epoch} took {time.time()-t:.2f}')
        print(f'loss_train : {loss_epoch_train}')
        print(f'loss_val : {loss_epoch_val}')
        print(f'acc_train : {acc_epoch_train} / topk_acc_train : {topk_acc_epoch_train}')
        print(f'acc_val : {acc_epoch_val} / topk_acc_val : {topk_acc_epoch_val} / '
              f'avgk_acc_val : {avgk_acc_epoch_val}')

        # Save logs at every epoch
        results = {'loss_train': loss_train,
                   'acc_train': acc_train,
                   'topk_acc_train': topk_acc_train,
                   'loss_val': loss_val,
                   'acc_val': acc_val,
                   'topk_acc_val': topk_acc_val,
                   'class_acc_val': class_acc_val,
                   'avgk_acc_val': avgk_acc_val,
                   'macro_topk_acc_val': macro_topk_acc_val,
                   'lmbda_best_acc': lmbda_best_acc,
                   'class_to_idx': dataset_attributes['class_to_idx'],
                   'params': configs}

        with open(os.path.join(save_dir, save_name + '_train_val.pkl'), 'wb') as f:
            pickle.dump(results, f)

    writer.close()


