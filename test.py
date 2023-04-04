import os
import pickle
import torch
from torch.nn import CrossEntropyLoss
from utils import set_seed, load_model, get_model
from epoch import test_epoch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def test(args, test_loader, dataset_attributes):

    if not args.test:
        return

    set_seed(args, use_gpu=torch.cuda.is_available())

    results = []
    with (open(f'{args.results_dir}/small_dataset_train_val.pkl', "rb")) as openfile:
        while True:
            try:
                results.append(pickle.load(openfile))
            except EOFError:
                break

    lmbda_best_acc = results[0]['lmbda_best_acc']

    model = get_model(args, n_classes=dataset_attributes['n_classes'])

    criteria = CrossEntropyLoss()
    load_model(model, args.best_model, args.use_gpu)

    loss_test_ba, acc_test_ba, topk_acc_test_ba, \
    avgk_acc_test_ba, class_acc_test = test_epoch(model, test_loader, criteria, args.k,
                                                  lmbda_best_acc, args.use_gpu,
                                                  dataset_attributes)

    # Save Test results as a dictionary and save it as a pickle file in desired location
    results = {'test_results': {'loss': loss_test_ba,
                                'accuracy': acc_test_ba,
                                'topk_accuracy': topk_acc_test_ba,
                                'avgk_accuracy': avgk_acc_test_ba,
                                'class_acc_dict': class_acc_test},
               'params': args.__dict__}

    with open(f'{args.results_dir}/small_dataset_test.pkl', 'wb') as f:
        pickle.dump(results, f)


