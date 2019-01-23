import os
import numpy as np

label_idx = 7

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
    elif dataset == 'test':
        file_name = 'test_dataset.txt'
    else:
        print("Unrecognized filename.")
        return

    file_path = os.path.join('wifi_db', file_name)
    
    data = np.loadtxt(file_path)
    # print(data)
    return data

def decision_tree_learning(dataset, depth):
    if same_label(dataset):
        return (dataset, depth)
    else:
        (s_attr, s_val, l_dataset, r_dataset) = find_split(dataset)
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)

        node = {'attr': s_attr,
                'val': s_val,
                'left': l_branch,
                'right': r_branch}
                
        return (node, max(l_depth, r_depth))
        
def find_split(dataset):
    pass

def info_gain(l_dataset, r_dataset):
    pass

'''
Verify if all labels in the dataset are the same:
if a label different from the first label appears,
exit early without checking the rest.
'''
def same_label(dataset):
    comp = dataset[0][label_idx]
    for data in dataset:
        if data[label_idx] != comp:
            return False

    return True


# d = load_data('test')
# print(d)
# print(same_label(d))