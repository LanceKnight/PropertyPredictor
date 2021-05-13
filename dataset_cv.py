'''

running this .py file directly will generate a .cfg file that has all the training, validation, test indices

usage: python dataset_cv.py [col]

[col] is the column to split

'''

import torch
from random import sample, seed
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader, InMemoryDataset, Data
from sklearn.model_selection import StratifiedKFold
from configparser import SafeConfigParser
from tqdm import tqdm
import numpy as np
try: 
	from rdkit import Chem
	from rdkit import DataStructs
except ImportError:
	Chem = None
	DataStructs = None

#from printing import tee_print
import printing
import sys
import os.path as osp
import re

from molecule_processing import smiles2graph



class Sider(InMemoryDataset):
	names = {
		'sider':'ori_sider',
		'sider_toxcast':'sider_toxcast',
		'sider_toxcast_tox21':'sider_toxcast_tox21',
		'sider_toxcast_tox21_pcba':'sider_toxcast_tox21_pbca',
		'sider_pcba':'sider_pbca'
	}

	def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
		super(Sider, self).__init__(root, transform, pre_transform, pre_filter=None)
		self.name = name.lower()
		assert self.name in self.names.keys()
		self.data, self.slices = torch.load(self.processed_paths[0])
		self.similarity_matrix = torch.load(self.processed_paths[1])		
	@property
	def raw_file_names(self):
		return 'sider.csv'#f'{self.names[self.name]}.csv'#
	@property
	def processed_file_names(self):
		return ['data.pt', 'similarity.pt']
	
	@property
	def raw_dir(self):
		return osp.join(self.root, 'raw')

	@property
	def processed_dir(self):
		return osp.join(self.root, 'processed')

	def process(self):
		fp_lst = []
		with open(self.raw_paths[0], 'r') as f:
			dataset = f.read().split('\n')[1:-1]
			dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.
			self.similarity_matrix = np.zeros((len(dataset), len(dataset)))
			data_list = []
			for i, line in enumerate(tqdm(dataset)):
				line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
				line = line.split(',')
				
				smiles = line[0]
				ys = line[slice(1,28)]
				ys = ys if isinstance(ys, list) else [ys]
				
				ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
				y = torch.tensor(ys, dtype=torch.float).view(1, -1)
				
				mol = Chem.MolFromSmiles(smiles)
				if mol is None:
					continue
				
				
				x, edge_attr, edge_index = smiles2graph(smiles, molecular_attributes= True)
				# Sort indices.
				#if edge_index.numel() > 0:
				#	perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
				#	edge_index  = edge_index[:, perm]
				
				#print(f"=========")
				#print(f"smiles = {smiles}"	)
				#print(f"edge_index=\n{edge_index}")
				#print(f"edge_attr=\n")
				#for att in edge_attr:
				#	print(att)

				data = Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_attr), y=torch.tensor(y), smiles=smiles, id = i)
				if self.pre_filter is not None and not self.pre_filter(data):
					continue
				
				if self.pre_transform is not None:
					data = self.pre_transform(data)
				
				data_list.append(data)
				
				fp = Chem.RDKFingerprint(mol)
				fp_lst.append(fp)
				for j, past_fp in enumerate(fp_lst[:-1]):
					similarity = DataStructs.FingerprintSimilarity(fp, past_fp)
					self.similarity_matrix[i][j] = self.similarity_matrix[j][i]= similarity
					#print(f"smi_sc:{similarity}")
			self.similarity_matrix = torch.tensor(self.similarity_matrix)
			torch.save(self.collate(data_list), self.processed_paths[0])
			torch.save(self.similarity_matrix, self.processed_paths[1])
		#print(self.similarity_matrix)	


#SIDER = MoleculeNet(root = "data/SIDER", name = "SIDER")
SIDER = Sider(root = "data/SIDER/sider", name = 'sider_pcba')
#print(type(SIDER.similarity_matrix))
#SIDER = MoleculeNet(root = "/home/liuy69/.clearml/venvs-builds/3.6/task_repository/PropertyPredictor.git/side_effect/final_model/data", name = "SIDER")# This is a combined dataset, the first 1427 samples are labeld from SIDER. Then 8597 sampes from ToxCast (19 of them were discarded due to the failure to convert to mol), 7831 samples were from Tox21. The total number of samples are 1427+8597+7831-19 = 17836

NUM_LABELED = 1427

num_folds = 5
data_split_file = 'data_split_idx.cfg'

print(f"num of data:{len(SIDER)}, {NUM_LABELED} of them are labeled")
#print("sample data:")
#s = SIDER[0]
#print(s)
#print(f"smi:{s.smiles}\nx:\n{s.x}\n edge_index:\n{s.edge_index}\n edge_attr:{s.edge_attr}")

num_samples = len(SIDER)

train_dataset =[]
val_dataset = []
test_dataset =[]
train_num = 0

	


def get_loaders_with_idx(num_extra_data, batch_size, fold, sample_seed = 1, torch_seed = 1):
	global train_dataset
	global val_dataset
	global test_dataset 
	global train_num

	# set seeds for random.sample and PyTorch seed
	seed(a=sample_seed)
	if torch_seed is not None:
		torch.manual_seed(torch_seed)
	else:
		torch.seed()


	ori_train_idx = get_idx(fold, 'train_idx')
	print(f"num_total_samples:{num_samples}, available_unlabled_samples:{num_samples-NUM_LABELED}, num_unlabled_samples_in_use:{num_extra_data}")
	extra_unlabeled_idx = sample(range(NUM_LABELED, num_samples), num_extra_data)
	train_idx = ori_train_idx + extra_unlabeled_idx
	#print(f"extra_unlabled_idx:{extra_unlabeled_idx}")	
	val_idx = get_idx(fold, 'validation_idx')
	test_idx = get_idx('test', 'test')


	ori_train_num = len(ori_train_idx)
	val_num = len(val_idx)
	test_num = len(test_idx)
	train_num = len(train_idx)

	#print(f"fold:{fold}, ori_train_num:{ori_train_num}, val_num:{val_num}, test_num:{test_num}, num_extra_data:{len(extra_unlabeled_idx)}, train_num:{train_num}") 
	#print(f"min_max_train {min(train_idx)}-{max(train_idx)},   min_max_val:{min(val_idx)}-{max(val_idx)},   min_max_test:{min(test_idx)}-{max(test_idx)}")

	train_dataset = Subset(SIDER, train_idx)
	val_dataset = Subset(SIDER, val_idx)
	test_dataset = Subset(SIDER, test_idx)

	print(f"fold:{fold}, ori_train_num {ori_train_num}, val_num:{len(val_dataset)}, test_num:{len(test_dataset)}, num_extra_data:{len(extra_unlabeled_idx)}, train_num:{len(train_dataset)}") 

	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = DataLoader(val_dataset, batch_size = val_num, shuffle = True)
	test_loader = DataLoader(test_dataset, batch_size = test_num, shuffle = True)
	return train_loader, val_loader, test_loader


def get_loaders(num_extra_data, batch_size):
	# === 8:1:1 random split
	global train_dataset
	global val_dataset
	global test_dataset 
	global train_num
	
	all_idx = [x for x in range(NUM_LABELED)]
	ori_train_idx, test_idx = train_test_split(all_idx, test_size = 0.1)
	ori_train_idx, val_idx = train_test_split(ori_train_idx, test_size = 1.0/9)
	extra_unlabeled_idx = sample(range(NUM_LABELED, num_samples), num_extra_data)
	train_idx = ori_train_idx + extra_unlabeled_idx
	ori_train_num = len(ori_train_idx)
	val_num = len(val_idx)
	test_num = len(test_idx)
	train_num = len(train_idx)

	#print(f"min_max_train {min(train_idx)}-{max(train_idx)},   min_max_val:{min(val_idx)}-{max(val_idx)},   min_max_test:{min(test_idx)}-{max(test_idx)}")

	train_dataset = Subset(SIDER, train_idx)#SIDER[:ori_train_num] + SIDER[1427:(1427+num_extra_data)]
	val_dataset = Subset(SIDER, val_idx)#SIDER[val_idx]#SIDER[ori_train_num:ori_train_num+val_num]
	test_dataset = Subset(SIDER, test_idx)#SIDER[test_idx]#SIDER[1427-test_num:1427]

	print(f"ori_train_num {ori_train_num}, val_num:{len(val_dataset)}, test_num:{len(test_dataset)}, num_extra_data:{len(extra_unlabeled_idx)}, train_num:{len(train_dataset)}") 

	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = DataLoader(val_dataset, batch_size = val_num, shuffle = True)
	test_loader = DataLoader(test_dataset, batch_size = test_num, shuffle = True)
	return train_loader, val_loader, test_loader

def get_stats(col):
	'''
	Get the number of unlabeled
	'''
	global train_dataset
	global val_dataset
	global test_dataset 
	global train_num

	num_labels = sum([~torch.isnan(x.y[:,col]) for x in train_dataset]).item()
	num_unlabeled = sum([torch.isnan(x.y[:,col]) for x in train_dataset]).item()
	num_train = train_num
	return num_train, num_labels, num_unlabeled

def generate_idx(col):
	all_idx = [x for x in range(NUM_LABELED)]

	ori_train_idx, test_idx = train_test_split(all_idx, test_size = 0.1)
	set_idx('test', 'test_idx', str(test_idx))

	ori_train_dataset = SIDER[ori_train_idx]
	skf = StratifiedKFold(n_splits=5, shuffle=False)
	
	y =[data.y[:,0] for data in ori_train_dataset]

	folds = skf.split(ori_train_idx, y)
	
	fold_name = 'fold'
	for i, fold in enumerate(folds):
		section = fold_name + str(i+1)
		train_idx = list(fold[0])
		validation_idx = list(fold[1])
		set_idx(section, 'train_idx', str(train_idx))
		set_idx(section, 'validation_idx', str(validation_idx)) 


def set_idx(section, option, value):
	file_name = data_split_file
	config_parser = SafeConfigParser()
	config_parser.read(file_name)

	if section not in config_parser.sections():
		config_parser.add_section(section)
	config_parser.set(section, option, value)
		
	with open(file_name, 'w') as f:
		config_parser.write(f)


def get_idx(section, option):
	file_name = data_split_file
	config_parser = SafeConfigParser()
	with open(file_name, 'r') as f:
		config_parser.read_file(f)
	if section =='test':
		return str2list(config_parser[section]['test_idx'])
	else:
		fold_name = 'fold'
		new_section = fold_name + str(section)
		return str2list(config_parser[new_section][option])

def str2list(input_str):
	output = input_str.strip('][').split(', ')
	output = list(map(int, output))
	return output


if __name__ == "__main__":

	col = sys.argv[1] # the column that needs the split
	generate_idx(col)
	print(f"generated indices for column {col}, stored in {data_split_file}")
	#get_loaders_with_idx(1, 64, 1)
