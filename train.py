import os
from tqdm import tqdm
import pickle
import time
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from utils import set_seed, load_model, save, get_model, update_optimizer, make_directory
from epoch import train_epoch, val_epoch, test_epoch
from torch.utils.tensorboard import SummaryWriter
from utils import MulticlassFocalLoss
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(args, train_loader, val_loader, dataset_attributes):

    if not args.train:
        return

    set_seed(args, use_gpu=torch.cuda.is_available())
    save_dir = args.results_dir
    save_name = args.exp_name

    writer = SummaryWriter(args.log + "/")

    model = get_model(args, n_classes=dataset_attributes['n_classes'])

    criteria = CrossEntropyLoss()

    if args.loss == "Focal":
        criteria = MulticlassFocalLoss(gamma=2)

    if args.use_gpu:
        print('USING GPU')
        torch.cuda.set_device(0)
        model.cuda()
        criteria.cuda()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.mu, nesterov=True)

    # Containers for storing metrics over epochs
    loss_train, acc_train, topk_acc_train = [], [], []
    loss_val, acc_val, topk_acc_val, avgk_acc_val, class_acc_val, macro_topk_acc_val = [], [], [], [], [], []

    print('args.k : ', args.k)

    lmbda_best_acc = None
    best_val_acc = float('-inf')

    for epoch in tqdm(range(args.n_epochs), desc='epoch', position=0):

        t = time.time()
        optimizer = update_optimizer(optimizer, lr_schedule=args.epoch_decay, epoch=epoch)

        # #### Training
        loss_epoch_train, acc_epoch_train, topk_acc_epoch_train = train_epoch(model, optimizer, train_loader,
                                                                              criteria, loss_train, acc_train,
                                                                              topk_acc_train, args.k,
                                                                              dataset_attributes['n_train'],
                                                                              args.use_gpu)

        writer.add_scalar('loss_epoch_train', loss_epoch_train, epoch)
        writer.add_scalar('acc_epoch_train', acc_epoch_train, epoch)
        for k in args.k:
            writer.add_scalar(f'topk_acc_epoch_train-Top{k}:', topk_acc_epoch_train[k], epoch)

        # #### Validation
        loss_epoch_val, acc_epoch_val, topk_acc_epoch_val, \
        avgk_acc_epoch_val, macro_topk_acc_epoch_val, lmbda_val = val_epoch(model, val_loader, criteria,
                                                                            loss_val, acc_val, topk_acc_val, avgk_acc_val,
                                                                            class_acc_val, macro_topk_acc_val,
                                                                            args.k, dataset_attributes, args.use_gpu)

        writer.add_scalar('loss_epoch_val', loss_epoch_val, epoch)
        writer.add_scalar('acc_epoch_val', acc_epoch_val, epoch)
        for k in args.k:
            writer.add_scalar(f'avgk_acc_epoch_val-Top{k}:', avgk_acc_epoch_val[k], epoch)
            writer.add_scalar(f'lmbda_val-Top{k}:', lmbda_val[k], epoch)
            writer.add_scalar(f'topk_acc_epoch_val-Top{k}:', topk_acc_epoch_val[k], epoch)

        # save model at every epoch
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


    # Save Train and Validation results as a dictionary and save it as a pickle file in desired location
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
               'params': args.__dict__}

    with open(os.path.join(save_dir, save_name + '_train_val.pkl'), 'wb') as f:
        pickle.dump(results, f)

    writer.close()


