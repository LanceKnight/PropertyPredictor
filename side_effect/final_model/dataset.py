import torch
from random import sample
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader


SIDER = MoleculeNet(root = "../../data/raw/SIDER", name = "SIDER")# This is a combined dataset, the first 1427 samples are labeld from SIDER. Then 8597 sampes from ToxCast (19 of them were discarded due to the failure to convert to mol), 7831 samples were from Tox21. The total number of samples are 1427+8597+7831-19 = 17836

NUM_LABELED = 1427

print(f"num of data:{len(SIDER)}, {NUM_LABELED} of them are labeled")

num_samples = len(SIDER)
num_extra_data = 0#8000
all_idx = [x for x in range(NUM_LABELED)]
ori_train_idx, test_idx = train_test_split(all_idx, test_size = 0.1)
ori_train_idx, val_idx = train_test_split(ori_train_idx, test_size = 1.0/9)
extra_unlabeled_idx = sample(range(1427, num_samples), num_extra_data)
train_idx = ori_train_idx + extra_unlabeled_idx
ori_train_num = len(ori_train_idx)
val_num = len(val_idx)
test_num = len(test_idx)
train_num = len(train_idx)

print(f"ori_train_num {ori_train_num}, val_num:{val_num}, test_num:{test_num}, num_extra_data:{len(extra_unlabeled_idx)}, train_num:{train_num}") 
print(f"min_max_train {min(train_idx)}-{max(train_idx)},   min_max_val:{min(val_idx)}-{max(val_idx)},   min_max_test:{min(test_idx)}-{max(test_idx)}")

train_dataset = Subset(SIDER, train_idx)#SIDER[:ori_train_num] + SIDER[1427:(1427+num_extra_data)]
val_dataset = Subset(SIDER, val_idx)#SIDER[val_idx]#SIDER[ori_train_num:ori_train_num+val_num]
test_dataset = Subset(SIDER, test_idx)#SIDER[test_idx]#SIDER[1427-test_num:1427]

def Get_Loaders(batch_size):
	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = DataLoader(val_dataset, batch_size = val_num, shuffle = True)
	test_loader = DataLoader(test_dataset, batch_size = test_num, shuffle = True)
	return train_loader, val_loader, test_loader

def Get_Stats(col):
	'''
	Get the number of unlabeled
	'''
	num_labels = sum([~torch.isnan(x.y[:,col]) for x in train_dataset]).item()
	num_unlabeled = sum([torch.isnan(x.y[:,col]) for x in train_dataset]).item()
	num_train = train_num
	return num_train, num_labels, num_unlabeled
