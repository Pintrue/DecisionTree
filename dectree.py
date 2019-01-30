import os
import numpy as np
import functools as ft
import math
import random
import matplotlib.pyplot as plt
# import pydot

WIFI_NUM = 7
LABEL_IDX = WIFI_NUM
LABEL_NUM = 4
LABEL_START = 1
LABEL_END = 5

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
def find_column_split(dataset, wifi):
	last_data = dataset[0]
	s_left = []
	s_right = dataset
	s_left.append(s_right.pop(0))

	info_gains = []

	for i in range(len(s_right)):
		if last_data[LABEL_IDX] != s_right[0][LABEL_IDX]:
			if last_data[wifi] != s_right[0][wifi]:
				split_val = (last_data[wifi] + s_right[0][wifi]) / 2.0
				info_gains.append((i + 1, info_gain(s_left, s_right), split_val))

		last_data = s_right[0]
		s_left.append(s_right.pop(0))

	max_info_gain = -float("INF")

	if len(info_gains) == 0:
		# print("Cannot find a split for attribute %d" % wifi)
		return (0, max_info_gain, 0)

	max_tuple = None
	for ig in info_gains:
		if ig[1] > max_info_gain:
			max_info_gain = ig[1]
			max_tuple = ig

	return max_tuple


def find_all_col_split(dataset, wifi):
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
	return max_tuple


def best_split(dataset, split_func):
	info_gains = []
	for i in range(0, WIFI_NUM):
		sorted_dataset = sorted(dataset, \
								key=ft.cmp_to_key( \
									lambda x, y: cmp_data_tuple(x, y, i)))
		tp = split_func(sorted_dataset, i)
		if (tp != None) and (tp[1] != -float("INF")):
			info_gains.append((i,) + tp + (sorted_dataset,))

	max_info_gain = -float("INF")
	max_tuple = None

	for ig in info_gains:
		if ig[2] > max_info_gain:
			max_info_gain = ig[2]
			max_tuple = ig

	return max_tuple


def find_split(dataset):
	# max_tuple = best_split(dataset, find_all_col_split)
	max_tuple = best_split(dataset, find_column_split)

	# If no split is found when there is any difference
	# in labels, try to find any possible split values
	# with the highest information gains.
	if max_tuple == None:
		max_tuple = best_split(dataset, find_all_col_split)

	i = max_tuple[1]
	sorted_dataset = sorted(dataset, \
							key=ft.cmp_to_key( \
								lambda x, y: cmp_data_tuple(x, y, max_tuple[0])))
	return (max_tuple[0], max_tuple[3], sorted_dataset[:i], sorted_dataset[i:])


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
	l_dataset_len = float(len(l_dataset))
	r_dataset_len = float(len(r_dataset))
	dataset_len = l_dataset_len + r_dataset_len

	remainder = (l_dataset_len / dataset_len) * cal_entropy(l_dataset) + (r_dataset_len / dataset_len) * cal_entropy(
		r_dataset)

	return cal_entropy(l_dataset + r_dataset) - remainder


def cal_entropy(dataset):
	dataset_len = len(dataset)
	acc1 = 0
	acc2 = 0
	acc3 = 0
	acc4 = 0

	for index in range(dataset_len):
		if dataset[index][LABEL_IDX] == 1:
			acc1 += 1
		elif dataset[index][LABEL_IDX] == 2:
			acc2 += 1
		elif dataset[index][LABEL_IDX] == 3:
			acc3 += 1
		elif dataset[index][LABEL_IDX] == 4:
			acc4 += 1

	p1 = float(acc1) / dataset_len
	p2 = float(acc2) / dataset_len
	p3 = float(acc3) / dataset_len
	p4 = float(acc4) / dataset_len

	t1 = -p1 * math.log(p1, 2) if p1 > 0 else 0
	t2 = -p2 * math.log(p2, 2) if p2 > 0 else 0
	t3 = -p3 * math.log(p3, 2) if p3 > 0 else 0
	t4 = -p4 * math.log(p4, 2) if p4 > 0 else 0

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


'''
Validate the test set on the built decision tree,
return a tuple consisting of three elements:

wrong_num: number of incorrectly classified data
data_num: total number of data that have been classfied
wrong_set: set of the predicted and actual labels of
			all incorrectly classified data
'''
def evaluate(root, dataset):
	wrong_set = []
	correct_set = []
	for data in dataset:
		res = classify(root, data)
		if res != data[LABEL_IDX]:
			# wrong_set.append((data, res))
			wrong_set.append((int(data[LABEL_IDX]), int(res)))
		else:
			correct_set.append(int(res))
	data_num = len(dataset)
	wrong_num = len(wrong_set)
	return (wrong_num, data_num, wrong_set, correct_set)


'''
Cross validate the dataset, across the number of
fold it is separated into, which is specified by
the 'fold_num' argument.
'''
def cross_validation(dataset, fold_num):
	fold_len = int(len(dataset) / fold_num)
	cv_result = []
	confusion_mat = np.full((LABEL_NUM, LABEL_NUM), 0)

	for k in range(fold_num):
		validate_data = np.array(dataset[k * fold_len: (k + 1) * fold_len])
		train_data = np.array(dataset[: k * fold_len] + dataset[(k + 1) * fold_len:])

		tree = decision_tree_learning(train_data, 0)
		(wrong_num, _, wrong_set, correct_set) = evaluate(tree[0], validate_data)
		cv_result.append((k, wrong_num))

		print("Fold #%d has %d of wrongly labeled data, out of %d total data."
			  % (k, wrong_num, fold_len))

		for wrong in wrong_set:
			confusion_mat[wrong[0] - 1][wrong[1] - 1] += 1
		for correct in correct_set:
			confusion_mat[correct - 1][correct - 1] += 1

	avg_confmat = list(map(lambda l : list(map(lambda x : x / 10.0, l)), confusion_mat))
	#print(avg_confmat)
	confusion_mat = np.array(avg_confmat, dtype=np.float32)
	print(confusion_mat)

	cal_avg_accuracy(confusion_mat)
	plot_cm(confusion_mat)

	return (cv_result, confusion_mat)


def cross_validation_prune(dataset, fold_num):
	fold_len = int(len(dataset) / fold_num)
	# cv_result = []
	# test_fold_index = random.randint(0, fold_num - 1)
	for test_fold_index in range(fold_num):
		test_data = np.array(dataset[test_fold_index * fold_len : (test_fold_index + 1) * fold_len])
		rest_data = dataset[: test_fold_index * fold_len] + \
						dataset[(test_fold_index + 1) * fold_len :]

		for k in range(fold_num - 1):
			validate_data = np.array(rest_data[k * fold_len : (k + 1) * fold_len])
			train_data = np.array(rest_data[: k * fold_len] + rest_data[(k + 1) * fold_len :])

			tree = decision_tree_learning(train_data, 0)
			(wrong_num1, _, wrong_set1, correct_set1) = evaluate(tree[0], test_data)
			pruned_t = prune(tree[0], tree[0], validate_data)
			(wrong_num2, _, wrong_set2, correct_set2) = evaluate(tree[0], test_data)

			print(("Fold #%d has %d of wrong before pruning, " + \
			"%d after pruning, out of %d total data.")
					% (k, wrong_num1, wrong_num2, fold_len))
			# cv_result.append((k, wrong_num))

			# print("Fold #%d has %d of wrongly labeled data, out of %d total data."
			# 	  % (k, wrong_num, fold_len))


def prune(node, tree, validate_data):
	if node['leaf'] == True:
		return

	l_branch = node['left']
	r_branch = node['right']
	prune(l_branch, tree, validate_data)
	prune(r_branch, tree, validate_data)

	# if both branches are leaves, PRUNE.
	if l_branch['leaf'] and r_branch['leaf']:
		no_prune = evaluate(tree, validate_data)[0]
		node['leaf'] = True
		node['room'] = l_branch['room']
		prune_to_l = evaluate(tree, validate_data)[0]
		node['room'] = r_branch['room']
		prune_to_r = evaluate(tree, validate_data)[0]
		if prune_to_l < prune_to_r:
			if prune_to_l < no_prune:
				node['room'] = l_branch['room']
				node['leaf'] = True
				# clear other fields, todo
			else:
				node['leaf'] = False
				del node['room']
		else: # prune_to_r <= prune_to_l
			if prune_to_r < no_prune:
				node['room'] = r_branch['room']
				node['leaf'] = True
				# clear other fields, todo
			else:
				node['leaf'] = False
				del node['room']




def metrics(confusion_mat, label):
	tp = confusion_mat[label - 1][label - 1]
	fp = np.sum(confusion_mat, axis=0)[label - 1] - tp
	fn = np.sum(confusion_mat, axis=1)[label - 1] - tp
	tn = confusion_mat.trace() - tp
	return (tp, fp, fn, tn)


def cal_avg_accuracy(confusion_mat):
	res = []
	for index in range(LABEL_START, LABEL_END):
		(tp, fp, fn, tn) = metrics(confusion_mat, index)
		recall = tp / (tp + fn)
		precision = tp / (tp + fp)
		class_rate = (tp + tn) / (tp + tn + fp + fn)
		f1_ms = 2 * precision * recall / (precision + recall)
		print(recall, precision, class_rate, f1_ms)
		res.append((index, recall, precision, class_rate, f1_ms))

	return res


def plot_cm(confusion_mat):
	plt.imshow(confusion_mat, cmap=plt.cm.Blues)
	classNames = ['Room 1', 'Room 2', 'Room 3', 'Room 4']
	plt.title('Confusion Matrix - Average Cross Validation Classification Results')
	plt.ylabel('Actual label')
	plt.xlabel('Predicted label')
	tick_marks = np.arange(len(classNames))
	plt.xticks(tick_marks, classNames, rotation=45)
	plt.yticks(tick_marks, classNames)
	plt.show()


'''
Randomly shuffle the original dataset,
which does not mutate the original dataset.
Return the shuffled data in LIST.
'''
def shuffle_data(dataset):
	shuffled = random.sample(dataset.tolist(), len(dataset))
	return shuffled


# def draw(parent_name, child_name):
# 	edge = pydot.Edge(parent_name, child_name)
# 	graph.add_edge(edge)
#
#
# def visit(node, parent=None):
# 	if node['leaf'] == False:
# 		if parent:
# 			draw(parent, str(node['attr']) + ' | %f' % node['val'])
# 		visit(node['left'], str(node['attr']) + ' | %f' % node['val'])
# 		visit(node['right'], str(node['attr']) + ' | %f' % node['val'])
# 	else:
# 		draw(parent, 'Room %f' % node['room'])


'''
Main program starts here
'''
d = load_data('noisy')

# t = decision_tree_learning(d, 0)

# tst = load_data('clean')
# (w, t) = validate(t[0], tst)
# print("%d wrongly labeled, out of %d test data." % (w, t))

# 10-fold cross validation
shuffled_data = shuffle_data(d)
cross_validation_prune(shuffled_data, 10)

# graph = pydot.Dot(graph_type='graph')
# visit(t[0])
# graph.write_png("1.png")

# print(t)
