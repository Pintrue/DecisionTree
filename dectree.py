import os
import numpy as np

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


def info_gain(all, left, right):
    pass


# load_data('clean')