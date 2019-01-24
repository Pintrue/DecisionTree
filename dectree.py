import os
import numpy as np
import functools as ft
import math
import pydot

WIFI_NUM = 7
LABEL_IDX = WIFI_NUM

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
    return data

def decision_tree_learning(dataset, depth):
    if same_label(dataset):
        node = {'leaf': True,
                'room': dataset[0][LABEL_IDX]}
        return (node, depth)
    else:
        (s_attr, s_val, l_dataset, r_dataset) = find_split(dataset)
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)

        node = {'attr': s_attr,
                'val': s_val,
                'left': l_branch,
                'right': r_branch,
                'leaf': False}

        return (node, max(l_depth, r_depth))

def cmp_data_tuple(t1, t2, wifi):
	return t1[wifi] - t2[wifi]

'''
pre: the wifi column is sorted, dataset must be N * 8
post: dataset content might be changed
return: (splitidx, infogain, splitval)
'''
def find_best_split(dataset, wifi):
	last = dataset[0][wifi]
	last_label = dataset[0][LABEL_IDX]
	sleft = []
	sright = dataset
	info_gains = []
	i = 0
	while len(sright) > 0:
		t = sright[0]
		if t[wifi] != last or t[LABEL_IDX] != last_label:
			splitval = (t[wifi] + last) / 2.0
			info_gains.append((i, info_gain(sleft, sright), splitval))
			last = t[wifi]
			last_label = t[LABEL_IDX]
		sleft.append(sright.pop(0))
		i += 1
	max_info_gain = -float("INF")
	max_tuple = None
	for ig in info_gains:
		if ig[1] > max_info_gain:
			max_info_gain = ig[1]
			max_tuple = ig
	return max_tuple

#attribute, value, sleft, sright
def find_split(dataset):
	info_gains = []
	for i in range(0, WIFI_NUM):
		sorted_dataset = sorted(dataset, \
			key=ft.cmp_to_key( \
			lambda x,y : cmp_data_tuple(x, y, i)))
		tp = find_best_split(sorted_dataset, i)
		info_gains.append((i,) + tp)
	max_info_gain = -float("INF")
	max_tuple = None
	for ig in info_gains:
		if ig[2] > max_info_gain:
			max_info_gain = ig[2]
			max_tuple = ig
	i = max_tuple[1]
	return (max_tuple[0],max_tuple[3],dataset[:i],dataset[i:])

'''
Verify if all labels in the dataset are the same:
if a label different from the first label appears,
exit early without checking the rest.
'''
def same_label(dataset):
    if len(dataset) == 0:
        return True

    comp = dataset[0][LABEL_IDX]
    for data in dataset:
        if data[LABEL_IDX] != comp:
            return False

    return True

def info_gain(l_dataset, r_dataset):
    l_dataset_len = len(l_dataset)
    r_dataset_len = len(r_dataset)
    dataset_len = l_dataset_len + r_dataset_len

    remainder = (l_dataset_len / dataset_len) * cal_entropy(l_dataset) + (r_dataset_len / dataset_len) * cal_entropy(r_dataset)

    return cal_entropy(l_dataset + r_dataset) - remainder


def cal_entropy(dataset):
    dataset_len = len(dataset)
    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 = 0

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

def classify(node, data):
	if node['leaf'] == True:
		return node['room']
	else:
		attr, val, l, r, _ = node.values()
		v = data[attr]
		if v < val:
			return classify(l, data)
		else:
			return classify(r, data)

def evaluate(node, dataset):
	wrong_set = []
	for data in dataset:
		res = classify(node, data)
		if res != data[LABEL_IDX]:
			wrong_set.append((data, res))
	data_num = len(dataset)
	wrong_num = len(wrong_set)
	return (wrong_num, data_num)

def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    if node['leaf'] == False:
        if parent:
            draw(parent, str(node['attr']) + ' | %f' % node['val'])
        visit(node['left'], str(node['attr']) + ' | %f' % node['val'])
        visit(node['right'], str(node['attr']) + ' | %f' % node['val'])
    else:
        draw(parent, 'Room %f' % node['room'])

d = load_data('clean')
# print(d)
# print(same_label(d))
t = decision_tree_learning(d, 0)

tst = load_data('noisy')
(w, t) = evaluate(t[0], tst)
print("%d wrongly labeled, out of %d test data." % (w, t))

graph = pydot.Dot(graph_type='graph')
# visit(t[0])
# graph.write_png("1.png")

# print(t)