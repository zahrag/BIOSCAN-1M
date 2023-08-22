
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import seaborn as sns
from utils import make_directory


def vis_results(configs, results):

    res = results['test_results']
    print_results(configs, res)

    plot_results(res['class_acc_dict']['class_acc'], res['class_to_idx'],
                 dataset=configs['exp_name'],
                 metric='Micro Accuracy',
                 fig_path=f"{configs['results_path']}/figures",
                 plot=True)

    make_confusion_matrix(res['y_true'], res['y_pred'], res['class_to_idx'],
                          exp_name=configs['exp_name'],
                          fig_path=f"{configs['results_path']}/figures",
                          plot=True)


def print_results(configs, res):

    print(f" \n\n--------------- TEST RESULTS --------------- ")
    print(f"Experiment: {configs['exp_name']} Model:{configs['model']} Loss:{configs['loss']}")
    print(f"Top-K Accuracy: {res['topk_accuracy']}")
    print(f"Macro-Avg-Class Accuracy: {res['macro_topk_acc_test']}")
    print(f"Agerage Top-K Accuracy:{res['avgk_accuracy']}")
    print(f"F1-Score-Micro:{res['f1_score']['f1_micro']}\nF1-Score-Macro:{res['f1_score']['f1_macro']}")


def plot_results(class_acc, class_idx, dataset='', metric='', fig_path='', plot=False):

    if not plot:
        return

    class_names = list(class_idx.keys())
    micro = [class_acc[id] for id in class_idx.values()]
    N = len(micro)

    cmaps = ['Blues', 'YlGnBu']
    micro = np.reshape(micro, (1, N))
    deg_r = 45
    if micro.shape[1] > 20:
        deg_r = 75

    plt.figure(figsize=(8, 3))
    ax = sns.heatmap(micro, annot=True, cmap=cmaps[0], yticklabels=[metric], xticklabels=class_names,
                     annot_kws={'fontsize': 10, 'fontweight': 'bold'}, cbar=False)

    for t in ax.get_xticklabels():
        t.set_rotation(deg_r)
    for t in ax.texts:
        t.set_rotation(90)

    plt.title(f'Class-Micro Accuracy: {dataset} dataset', fontweight='bold', fontsize=12)
    plt.tight_layout()
    make_directory(fig_path)
    plt.savefig(f"{fig_path}/heatmap_class_{dataset}_test.png", dpi=300)
    plt.show()


def make_confusion_matrix(ytrue, ypred, class_to_idx, exp_name='', fig_path='', plot=False):

    if not plot:
        return

    y_true = ytrue.cpu().numpy().tolist()
    y_pred = torch.argmax(ypred, dim=-1).cpu().numpy().tolist()
    class_names = list(class_to_idx.keys())
    sorted_labels, class_names = descending_cm(y_true, y_pred, class_names)
    # Create a confusion matrix using the sorted classes
    cm = confusion_matrix(y_true, y_pred, labels=sorted_labels)

    plt.figure()
    plot_confusion_matrix(cm, classes=class_names,
                          normalize=True,
                          title=f'Confusion Matrix:{exp_name}, micro test Accuracy')

    make_directory(fig_path)
    plt.savefig(f"{fig_path}/{exp_name}_confusion_matrix.png", dpi=300)
    plt.show()


def descending_cm(y_true, y_pred, class_names):

    # Calculate the number of samples for each class
    labels, counts = np.unique(y_true, return_counts=True)
    label_counts = dict(zip(labels, counts))

    # Sort the classes in descending order based on the number of samples
    sorted_labels = [label for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)]

    class_names_sorted = []
    for ind in sorted_labels:
        class_names_sorted.append(class_names[ind])

    return sorted_labels, class_names_sorted


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Set the figure size
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(shrink=0.85)
    tick_marks = np.arange(len(classes))
    deg_r, f_size = 45, 10
    if len(classes) > 20:
        deg_r, f_size = 75, 5
    plt.xticks(tick_marks, classes, rotation=deg_r)
    plt.yticks(tick_marks, classes, rotation=0)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=f_size)

    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.tight_layout()
