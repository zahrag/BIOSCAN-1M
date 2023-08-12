import torch
from tqdm import tqdm
from utils import count_correct_topk, count_correct_avgk, update_correct_per_class, \
    update_correct_per_class_topk, update_correct_per_class_avgk

import torch.nn.functional as F
from collections import defaultdict
import itertools


def train_epoch(model, optimizer, train_loader, criteria, loss_train, acc_train, topk_acc_train, list_k, n_train, use_gpu):
    """Single train epoch pass. At the end of the epoch, updates the list loss_train, acc_train and topk_acc_train"""

    print("-------------- TRAIN ---------------")
    model.train()
    # Initialize variables
    loss_epoch_train = 0
    n_correct_train = 0
    # Containers for tracking nb of correctly classified examples
    # (in the top-k sense) and top-k accuracy for each k in list_k
    n_correct_topk_train = defaultdict(int)
    topk_acc_epoch_train = {}

    for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_loader, desc='train', position=0)):
        if use_gpu:
            batch_x_train, batch_y_train = batch_x_train.cuda(), batch_y_train.cuda()
        optimizer.zero_grad()
        batch_output_train = model(batch_x_train)

        loss_batch_train = criteria(batch_output_train, batch_y_train)
        loss_epoch_train += loss_batch_train.item()
        loss_batch_train.backward()
        optimizer.step()

        # Update variables
        with torch.no_grad():
            n_correct_train += torch.sum(torch.eq(batch_y_train, torch.argmax(batch_output_train, dim=-1))).item()
            for k in list_k:
                n_correct_topk_train[k] += count_correct_topk(scores=batch_output_train, labels=batch_y_train, k=k).item()

    # At the end of epoch compute average of statistics over batches and store them
    with torch.no_grad():
        loss_epoch_train /= batch_idx
        epoch_accuracy_train = n_correct_train / n_train
        for k in list_k:
            topk_acc_epoch_train[k] = n_correct_topk_train[k] / n_train

        loss_train.append(loss_epoch_train)
        acc_train.append(epoch_accuracy_train)
        topk_acc_train.append(topk_acc_epoch_train)

    return loss_epoch_train, epoch_accuracy_train, topk_acc_epoch_train


def val_epoch(model, val_loader, criteria, loss_val, acc_val, topk_acc_val, avgk_acc_val,
              class_acc_val, macro_topk_acc_val, list_k, dataset_attributes, use_gpu):
    """Single val epoch pass.
    At the end of the epoch, updates the lists loss_val, acc_val, topk_acc_val and avgk_acc_val"""

    print("\n-------------- VALIDATION ---------------")
    model.eval()
    with torch.no_grad():
        n_val = dataset_attributes['n_val']
        # Initialization of variables
        loss_epoch_val = 0
        n_correct_val = 0
        n_correct_topk_val, n_correct_avgk_val = defaultdict(int), defaultdict(int)
        topk_acc_epoch_val, avgk_acc_epoch_val, macro_topk_acc_epoch_val = {}, {}, {}
        # Store avg-k threshold for every k in list_k
        lmbda_val = {}
        # Store class accuracy, and top-k and average-k class accuracy for every k in list_k
        class_acc_dict = {}
        class_acc_dict['class_acc'] = defaultdict(int)
        class_acc_dict['class_topk_acc'], class_acc_dict['class_avgk_acc'] = {}, {}
        for k in list_k:
            class_acc_dict['class_topk_acc'][k], class_acc_dict['class_avgk_acc'][k] = defaultdict(int), defaultdict(int)

        # Store estimated probas and labels of the whole validation set to compute lambda
        list_val_proba, list_val_labels = [], []
        for batch_idx, (batch_x_val, batch_y_val) in enumerate(tqdm(val_loader, desc='val', position=0)):
            if use_gpu:
                batch_x_val, batch_y_val = batch_x_val.cuda(), batch_y_val.cuda()

            # Predicted labels as output of the model
            batch_output_val = model(batch_x_val)

            # Probability vector of all classes
            batch_proba = F.softmax(batch_output_val)

            # Store batch probas and labels
            list_val_proba.append(batch_proba)
            list_val_labels.append(batch_y_val)

            # Validation loss
            loss_batch_val = criteria(batch_output_val, batch_y_val)
            loss_epoch_val += loss_batch_val.item()

            # Accuracy over epoch
            n_correct_val += torch.sum(torch.eq(batch_y_val, torch.argmax(batch_output_val, dim=-1))).item()
            update_correct_per_class(batch_proba, batch_y_val, class_acc_dict['class_acc'])

            # Update top-k count and top-k count for each class
            for k in list_k:
                n_correct_topk_val[k] += count_correct_topk(scores=batch_output_val, labels=batch_y_val, k=k).item()
                update_correct_per_class_topk(batch_proba, batch_y_val, class_acc_dict['class_topk_acc'][k], k)

        """
        The new_app: if the flag is set to True then the lambda threshold values are calculated using 
        a new approach for BIOSCAN-1M, which is different from PlantNet-300K implementation.
        """
        new_app = True
        if new_app:
            val_probas = torch.cat(list_val_proba)  # ((n_batch*batch_size) , n_class)=(n_sample,n_class)
            sorted_sample_probas, _ = torch.sort(val_probas, dim=1, descending=True)

            a = [[] for n in range(sorted_sample_probas.shape[1])]
            for k in range(sorted_sample_probas.shape[1]):  # n_class
                a[k] = [sample[k] for sample in sorted_sample_probas]
            sorted_probas = list(itertools.chain(*a))
            sorted_probas = torch.tensor(sorted_probas)
        else:
            val_probas = torch.cat(list_val_proba)
            flat_val_probas = torch.flatten(val_probas)
            sorted_probas, _ = torch.sort(flat_val_probas, descending=True)

        val_labels = torch.cat(list_val_labels)   # ((n_batch*batch_size) , 1)
        for k in list_k:
            # Computes threshold for every k and count nb of correctly classifier examples in the avg-k sense
            # (globally and for each class)
            if new_app:
                lmbda_val[k] = 0.5 * (sorted_probas[n_val * (k - 1)] + sorted_probas[n_val * k - 1])
            else:
                lmbda_val[k] = 0.5 * (sorted_probas[n_val * k - 1] + sorted_probas[n_val * k])

            n_correct_avgk_val[k] += count_correct_avgk(probas=val_probas, labels=val_labels, lmbda=lmbda_val[k]).item()
            update_correct_per_class_avgk(val_probas, val_labels, class_acc_dict['class_avgk_acc'][k], lmbda_val[k])

        # After seeing val set update the statistics over batches and store them
        loss_epoch_val /= batch_idx
        epoch_accuracy_val = n_correct_val / n_val
        # Get top-k acc and avg-k acc
        for k in list_k:
            topk_acc_epoch_val[k] = n_correct_topk_val[k] / n_val
            avgk_acc_epoch_val[k] = n_correct_avgk_val[k] / n_val
        # Get class top-k acc and class avg-k acc
        for class_id in class_acc_dict['class_acc'].keys():
            n_class_val = dataset_attributes['class2num_instances']['val'][class_id]

            class_acc_dict['class_acc'][class_id] = class_acc_dict['class_acc'][class_id] / n_class_val
            for k in list_k:
                class_acc_dict['class_topk_acc'][k][class_id] = class_acc_dict['class_topk_acc'][k][class_id] / n_class_val
                class_acc_dict['class_avgk_acc'][k][class_id] = class_acc_dict['class_avgk_acc'][k][class_id] / n_class_val

        # Add Macro Acc
        for k in list_k:
            class_acc_topk_list = list(class_acc_dict['class_topk_acc'][k].values())
            macro_topk_acc_epoch_val[k] = sum(class_acc_topk_list) / dataset_attributes['n_classes']

        # Update containers with current epoch values
        loss_val.append(loss_epoch_val)
        acc_val.append(epoch_accuracy_val)
        topk_acc_val.append(topk_acc_epoch_val)
        avgk_acc_val.append(avgk_acc_epoch_val)
        class_acc_val.append(class_acc_dict)
        macro_topk_acc_val.append(macro_topk_acc_epoch_val)

    return loss_epoch_val, epoch_accuracy_val, topk_acc_epoch_val, avgk_acc_epoch_val, macro_topk_acc_epoch_val, lmbda_val


def test_epoch(model, test_loader, criteria, list_k, lmbda, use_gpu, dataset_attributes):

    print("\n-------------- TEST ---------------")
    model.eval()
    with torch.no_grad():
        n_test = dataset_attributes['n_test']
        loss_epoch_test = 0
        n_correct_test = 0
        topk_acc_epoch_test, avgk_acc_epoch_test, macro_topk_acc_epoch_test = {}, {}, {}
        n_correct_topk_test, n_correct_avgk_test = defaultdict(int), defaultdict(int)

        class_acc_dict = {}
        class_acc_dict['class_acc'] = defaultdict(int)
        class_acc_dict['class_topk_acc'], class_acc_dict['class_avgk_acc'] = {}, {}
        for k in list_k:
            class_acc_dict['class_topk_acc'][k], class_acc_dict['class_avgk_acc'][k] = defaultdict(int), defaultdict(int)

        list_test_proba, list_test_labels = [], []
        for batch_idx, (batch_x_test, batch_y_test) in enumerate(tqdm(test_loader, desc='test', position=0)):
            if use_gpu:
                batch_x_test, batch_y_test = batch_x_test.cuda(), batch_y_test.cuda()
            batch_output_test = model(batch_x_test)
            batch_proba_test = F.softmax(batch_output_test)
            loss_batch_test = criteria(batch_output_test, batch_y_test)
            loss_epoch_test += loss_batch_test.item()

            list_test_labels.append(batch_y_test)
            list_test_proba.append(batch_proba_test)

            n_correct_test += torch.sum(torch.eq(batch_y_test, torch.argmax(batch_output_test, dim=-1))).item()
            update_correct_per_class(batch_proba_test, batch_y_test, class_acc_dict['class_acc'])
            for k in list_k:
                n_correct_topk_test[k] += count_correct_topk(scores=batch_output_test, labels=batch_y_test, k=k).item()
                update_correct_per_class_topk(batch_output_test, batch_y_test, class_acc_dict['class_topk_acc'][k], k)
                if lmbda:
                    n_correct_avgk_test[k] += count_correct_avgk(probas=batch_proba_test, labels=batch_y_test, lmbda=lmbda[k]).item()
                    update_correct_per_class_avgk(batch_proba_test, batch_y_test, class_acc_dict['class_avgk_acc'][k], lmbda[k])

        y_true = torch.cat(list_test_labels)
        y_pred = torch.cat(list_test_proba)

        # After seeing test set update the statistics over batches and store them
        loss_epoch_test /= batch_idx
        acc_epoch_test = n_correct_test / n_test
        for k in list_k:
            topk_acc_epoch_test[k] = n_correct_topk_test[k] / n_test
            avgk_acc_epoch_test[k] = n_correct_avgk_test[k] / n_test

        for class_id in class_acc_dict['class_acc'].keys():
            n_class_test = dataset_attributes['class2num_instances']['test'][class_id]
            class_acc_dict['class_acc'][class_id] = class_acc_dict['class_acc'][class_id] / n_class_test
            for k in list_k:
                class_acc_dict['class_topk_acc'][k][class_id] = class_acc_dict['class_topk_acc'][k][class_id] / n_class_test
                class_acc_dict['class_avgk_acc'][k][class_id] = class_acc_dict['class_avgk_acc'][k][class_id] / n_class_test

        # Add Macro Acc
        for k in list_k:
            class_acc_topk_list = list(class_acc_dict['class_topk_acc'][k].values())
            macro_topk_acc_epoch_test[k] = sum(class_acc_topk_list) / dataset_attributes['n_classes']

    return loss_epoch_test, acc_epoch_test, topk_acc_epoch_test, avgk_acc_epoch_test, class_acc_dict, macro_topk_acc_epoch_test, y_true, y_pred
