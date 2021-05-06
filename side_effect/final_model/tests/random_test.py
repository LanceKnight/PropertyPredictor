import random
from random_dataset import get_loader
from dataset_cv import get_loaders_with_idx

num_extra_data = 0
batch_size = 10
fold=1

loader, _,_ = get_loaders_with_idx(num_extra_data, batch_size, fold)
for i, data in enumerate(loader):
	if(i==0):
		print(data.smiles[0])
		break
	 
