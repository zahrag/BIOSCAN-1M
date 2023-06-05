import os
import pickle
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_result_folder', type=str, default="results_for_comparing_using_original_and_cropped/small_res_ce_order_cropped_256_results")
    args = parser.parse_args()

    path_to_pkl = None
    path_to_config_txt = None
    for result_file_name in os.listdir(args.path_to_result_folder):
        if result_file_name.endswith('.pkl'):
            path_to_pkl = os.path.join(args.path_to_result_folder, result_file_name)
        if result_file_name.endswith('.txt'):
            path_to_config_txt = os.path.join(args.path_to_result_folder, result_file_name)

    with open(path_to_config_txt, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    config_dict = {}
    for i in lines:
        split_str = i.split(":")
        if len(split_str) < 2:
            config_dict[split_str[0]] = ""
        else:
            config_dict[split_str[0]] = split_str[1]

    # print(config_dict)

    str_1 = config_dict['model'] + " " + config_dict['loss'] + " " + config_dict['n_epochs'] + " " + config_dict[
        'lr'] + " " + config_dict['momentum'] + " " + config_dict['group_level']

    with open(path_to_pkl, 'rb') as f:
        data = pickle.load(f)


    top_1_acc_val = data['acc_val']
    epochs = range(1, len(top_1_acc_val) + 1)
    plt.plot(epochs, top_1_acc_val, 'b-', marker='o')

    plt.title('Top-1 accuracy on validation split over epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Top-1 accuracy on validation')
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0.8, 1)

    plt.savefig(os.path.join(args.path_to_result_folder, "val_top_1_acc_over_epoch.png"))

    plt.clf()

    train_loss = data['loss_train']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'b-', marker='o')

    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, 100)
    plt.ylim(0, 1)

    plt.grid(True)
    plt.savefig(os.path.join(args.path_to_result_folder, "loss_over_epoch.png"))

