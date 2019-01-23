import os
import numpy as np
import functools as ft

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


WIFI_NUM = 7
LABEL_IDX = WIFI_NUM

def cmp_data_tuple(t1, t2, wifi):
	return t1[wifi] - t2[wifi]

#pre: the wifi column is sorted, dataset must be N * 8
#post: dataset content might be changed
#return: (infogain, splitval, sleft, sright)
def find_best_split(dataset, wifi):
	last = dataset[0][wifi]
	sleft = []
	sright = dataset
	info_gains = []
	i = 0
	while len(sright) > 0:
		t = sright[0]
		if t[wifi] != last:
			splitval = (t[wifi] + last) / 2.0
			info_gains.append((i, info_gain(sleft, sright), splitval))
			last = t[wifi]
		sleft.append(sright.pop(0))
		i += 1
	max_info_gain = -float("INF")
	max_tuple = None
	for ig in info_gains:
		if ig[1] > max_info_gain:
			max_info_gain = ig[1]
			max_tuple = ig
	i = max_tuple[0]
	return max_tuple[1:] + (sleft[:i], sleft[i:])


#attribute, value, sleft, sright
def find_split(dataset):
	info_gains = []
	for i in xrange(0, WIFI_NUM):
		sorted_dataset = sorted(dataset, \
			key=ft.cmp_to_key( \
			lambda x,y : cmp_data_tuple(x, y, i)))
		tp = find_best_split(sorted_dataset, i)
		info_gains.append((i,) + tp)
	max_info_gain = -float("INF")
	max_tuple = None
	for ig in info_gains:
		if ig[1] > max_info_gain:
			max_info_gain = ig[1]
			max_tuple = ig
	return max_info_gain[0], max_info_gain[2:]

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
