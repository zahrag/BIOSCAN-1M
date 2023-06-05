import os
import pickle


def round_2_to_str(number):
    return str(round(number, 4))


def find_average(l):
    return sum(l) / len(l)


def list_2_string(l):
    result = ""
    for i in l:
        result = result + i + " "
    return result


if __name__ == '__main__':
    input_dir = "results_for_comparing_using_original_and_cropped/small_res_ce_order_cropped_256_results"
    path_to_pkl = None
    path_to_config_txt = None
    for result_file_name in os.listdir(input_dir):
        if result_file_name.endswith('.pkl'):
            path_to_pkl = os.path.join(input_dir, result_file_name)
        if result_file_name.endswith('.txt'):
            path_to_config_txt = os.path.join(input_dir, result_file_name)

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

    val_top_1 = round_2_to_str(data['topk_acc_val'][-1][1]) + " " + round_2_to_str(data['macro_topk_acc_val'][-1][1])
    val_top_5 = round_2_to_str(data['topk_acc_val'][-1][5]) + " " + round_2_to_str(data['macro_topk_acc_val'][-1][5])

    val_avg1 = find_average(data['class_acc_val'][-1]['class_avgk_acc'][1].values())
    val_avg5 = find_average(data['class_acc_val'][-1]['class_avgk_acc'][5].values())

    val_avg_1 = round_2_to_str(data['avgk_acc_val'][-1][1]) + " " + round_2_to_str(val_avg1)
    val_avg_5 = round_2_to_str(data['avgk_acc_val'][-1][5]) + " " + round_2_to_str(val_avg5)

    sorted_class_list = list(data['class_to_idx'].keys())
    sorted_class_list.sort()

    class_avg1_acc_val_dict = data['class_acc_val'][-1]['class_avgk_acc'][1]
    class_avg5_acc_val_dict = data['class_acc_val'][-1]['class_avgk_acc'][5]
    class_avg1_acc_val_list = []
    class_avg5_acc_val_list = []

    class_top1_acc_val_dict = data['class_acc_val'][-1]['class_topk_acc'][1]
    class_top5_acc_val_dict = data['class_acc_val'][-1]['class_topk_acc'][5]
    class_top1_acc_val_list = []
    class_top5_acc_val_list = []

    for class_name in sorted_class_list:
        curr_id = data['class_to_idx'][class_name]
        class_top1_acc_val_list.append(round_2_to_str(class_top1_acc_val_dict[curr_id]))
        class_top5_acc_val_list.append(round_2_to_str(class_top5_acc_val_dict[curr_id]))
        class_avg1_acc_val_list.append(round_2_to_str(class_avg1_acc_val_dict[curr_id]))
        class_avg5_acc_val_list.append(round_2_to_str(class_avg5_acc_val_dict[curr_id]))

    print(str_1)
    print(list_2_string(sorted_class_list))
    print(val_top_1 + " " + list_2_string(class_top1_acc_val_list))
    print(val_top_5 + " " + list_2_string(class_top5_acc_val_list))

    print(val_avg_1 + " " + list_2_string(class_avg1_acc_val_list))
    print(val_avg_5 + " " + list_2_string(class_avg5_acc_val_list))
