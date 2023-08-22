import os
import pickle
import torch
from torch.nn import CrossEntropyLoss
from utils import set_seed, load_model, get_model
from epoch import test_epoch
from utils import MulticlassFocalLoss
from visualize_results import vis_results
from utils import open_pickle
from utils import get_f1_score

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def test(configs, test_loader, dataset_attributes):

    if not configs["test"]:
        return

    set_seed(configs, use_gpu=torch.cuda.is_available())

    log_file = f'{configs["results_path"]}/{configs["best_model"]}_train_val.pkl'
    lmbda_best_acc = []
    if os.path.exists(log_file):
        tr_val_results = open_pickle(log_file)
        lmbda_best_acc = tr_val_results['lmbda_best_acc']
    else:
        print("To compute Average-K Accuracy:\nGet lmbda by testing the best trained model on the validation data!")

    best_model = f'{configs["results_path"]}/{configs["best_model"]}_weights_best_acc.tar'
    model = get_model(configs, n_classes=dataset_attributes['n_classes'])
    load_model(model, best_model, configs["use_gpu"])
    model.cuda()
    criteria = CrossEntropyLoss()
    if configs['loss'] == "Focal":
        criteria = MulticlassFocalLoss(gamma=2)

    loss_test_ba, acc_test_ba, topk_acc_test_ba, \
    avgk_acc_test_ba, class_acc_test, macro_topk_acc_test, y_true, y_pred = test_epoch(model, test_loader,
                                                                                       criteria, configs["k"],
                                                                                       lmbda_best_acc,
                                                                                       configs["use_gpu"],
                                                                                       dataset_attributes)

    # Calculate F1 scores
    f1_micro = get_f1_score(y_true, y_pred, metric='micro')
    f1_macro = get_f1_score(y_true, y_pred, metric='macro')

    # Save Test results as a dictionary and save it as a pickle file in desired location
    results = {'test_results': {'loss': loss_test_ba,
                                'accuracy': acc_test_ba,
                                'topk_accuracy': topk_acc_test_ba,
                                'avgk_accuracy': avgk_acc_test_ba,
                                'class_acc_dict': class_acc_test,
                                'macro_topk_acc_test': macro_topk_acc_test,
                                'y_true': y_true,
                                'y_pred': y_pred,
                                'class_to_idx': dataset_attributes['class_to_idx'],
                                'f1_score': {'f1_micro': f1_micro, 'f1_macro': f1_macro},
                                },
               'params': configs}

    # Visualize test results
    vis_results(configs, results)

    with open(f'{configs["results_path"]}/{configs["best_model"]}_test_results.pkl', 'wb') as f:
        pickle.dump(results, f)


