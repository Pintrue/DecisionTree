import os
import numpy as np
import math

'''
load_data/1 returns an array containing the data
from a file specified by argument dataset, which
can either be clean or noisy.
'''
def load_data(dataset):
    file_name = ''
    if dataset == 'clean':
        file_name = 'clean_dataset.txt'
    elif dataset == 'noisy':
        file_name = 'noisy_dataset.txt'
    file_path = os.path.join('wifi_db', file_name)

    data = np.loadtxt(file_path)
    # print(data)
    return data

def find_split(dataset):
    pass


def decision_tree_learning(training_dataset, depth):
    pass


def info_gain(l_dataset, r_dataset):
    l_dataset_len = l_dataset.shape[0]
    r_dataset_len = r_dataset.shape[0]
    dataset_len = l_dataset_len + r_dataset_len

    remainder = (l_dataset_len / dataset_len) * cal_entropy(l_dataset) + (r_dataset_len / dataset_len) * cal_entropy(r_dataset)

    return cal_entropy(l_dataset + r_dataset) - remainder


def cal_entropy(dataset):
    dataset_len = dataset.shape[0]
    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 =0

    for index in range(dataset_len):
        if dataset[index][7] == 1: acc1 += 1
        elif dataset[index][7] == 2: acc2 += 1
        elif dataset[index][7] == 3: acc3 += 1
        elif dataset[index][7] == 4: acc4 += 1

    p1 = acc1 / dataset_len
    p2 = acc2 / dataset_len
    p3 = acc3 / dataset_len
    p4 = acc4 / dataset_len

    t1 = -p1 * math.log(p1,2) if p1 > 0 else 0
    t2 = -p2 * math.log(p2,2) if p2 > 0 else 0
    t3 = -p3 * math.log(p3,2) if p3 > 0 else 0
    t4 = -p4 * math.log(p4,2) if p4 > 0 else 0

    return t1 + t2 + t3 + t4
