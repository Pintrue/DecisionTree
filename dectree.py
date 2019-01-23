import os
import numpy as np
import functools as ft
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


def decision_tree_learning(training_dataset, depth):
    pass


def info_gain(all, left, right):
    pass


# load_data('clean')