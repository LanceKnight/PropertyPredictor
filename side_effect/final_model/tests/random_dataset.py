from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader

loader=[]

def get_loader():
	global loader
	dataset = MoleculeNet(root='../../../data/raw/SIDER', name='SIDER')


	loader = DataLoader(dataset, batch_size = 5, shuffle = True)
	return loader
