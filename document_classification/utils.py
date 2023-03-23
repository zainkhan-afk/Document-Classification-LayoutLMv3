import glob
import os
import torch


def get_device(device = None):
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(device)

	return device

def get_paths(dataset_dir):
	label_names = os.listdir(dataset_dir)
	all_paths = glob.glob(dataset_dir+"/*/*.png")
	return all_paths, label_names

def load_paths_and_split(dataset_dir, val_split = 0.2):
	train_paths = []
	val_paths = []
	
	train_labels = []
	val_labels = []

	label_names = os.listdir(dataset_dir)
	idx = 0
	label_map = {}
	for label_name in label_names:

		label_map[idx] = label_name

		label_path = os.path.join(dataset_dir, label_name)
		all_img_paths = glob.glob(label_path + "/*.png")

		N = len(all_img_paths)
		val_N = int(val_split*N)

		train_paths += all_img_paths[val_N:]
		val_paths += all_img_paths[:val_N]


		train_labels += [idx]*(N-val_N)
		val_labels += [idx]*(val_N)

		idx += 1

	return train_paths, train_labels, val_paths, val_labels, label_map