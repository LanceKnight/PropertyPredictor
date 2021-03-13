import torch
import torch.nn.functional as F
from torch.nn import Linear, Tanh, Softmax, Sigmoid
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.data import DataLoader

import math
import numpy as np
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
from molecule_processing import batch2attributes, num_node_features, num_edge_features
 

num_epoches = 150
inner_atom_dim = 512
batch_size = 64
hidden_activation = Softmax()#Tanh()
conv_depth = 5

ESOL_dataset = MoleculeNet(root = "../data/raw/ESOL", name = "ESOL")
#print("data info:")
#print("============")
#print(f"num of data:{len(ESOL_dataset)}")

num_data = len(ESOL_dataset)
train_num = int(num_data * 0.8)
val_num = int(num_data * 0.0)
test_num = num_data - train_num - val_num
print(f"train_num = {train_num}, val_num = {val_num}, test_num = {test_num}")

train_loader = DataLoader(ESOL_dataset[:train_num], batch_size = batch_size, shuffle = True)
validate_loader = DataLoader(ESOL_dataset[train_num:train_num+val_num], batch_size = batch_size, shuffle = False)
test_loader = DataLoader(ESOL_dataset[-test_num:], batch_size = batch_size, shuffle = False)

class AtomBondConv(MessagePassing):
	def __init__(self, x_dim, edge_attr_dim):
		super(AtomBondConv, self).__init__(aggr = 'add')
		self.W_in = Linear(x_dim + edge_attr_dim, x_dim)

	def forward(self, x, edge_index, edge_attr, smiles, batch):		
		edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
		x = self.propagate(edge_index, x = x, edge_attr = edge_attr)
		x = self.W_in(x)
		return x

	def message(self, x, x_j, edge_attr):
		zero_tensor = torch.zeros(x.shape[0], edge_attr.shape[1], device = x_j.device) #create zero_tensor to pad the sec_col
		sec_col = torch.cat((edge_attr, zero_tensor), dim = 0) # create the second column. The first column has x_j, which is of shape (num_edge + num_node, num_node_feature), the second column has shape of (num_edge, edge_attr), padded with zero_tensor
		neighbor_atom_bond_feature = torch.cat((x_j, sec_col), dim = 1)
		return neighbor_atom_bond_feature


class MyNet(torch.nn.Module):
	def __init__(self, num_node_features, num_edge_features, depth):
		super(MyNet, self).__init__()
		self.atom_bond_conv = AtomBondConv(num_node_features, num_edge_features)
		self.W_out = Linear(num_node_features, inner_atom_dim)
		self.lin1 = Linear(inner_atom_dim, 50)
		self.lin2 = Linear(50, 1)
		self.depth = depth

	def forward(self, x, edge_index, edge_attr, smiles, batch):
		molecule_fp_lst = []
		for i in range(0, self.depth+1):
			#print(f"i:{i}")
			atom_fp = Softmax()(self.W_out(x))	
			molecule_fp = global_add_pool(atom_fp, batch)
			molecule_fp_lst.append(molecule_fp)
			x = self.atom_bond_conv(x, edge_index, edge_attr, smiles, batch)

		overall_molecule_fp	= torch.stack(molecule_fp_lst, dim=0).sum(dim=0)	
		hidden = Tanh()(self.lin1(overall_molecule_fp))
		out = self.lin2(hidden)
		return out
		
is_cuda = torch.cuda.is_available()
#print(f"is_cuda:{is_cuda}")

device = torch.device('cuda' if is_cuda else 'cpu')
model = MyNet(num_node_features, num_edge_features, conv_depth).to(device)
criterion = torch.nn.MSELoss()

#example
#data_loader = DataLoader(ESOL_dataset[0:1], batch_size = 1, shuffle= False)
#data = ESOL_dataset[12].to(device)
#print(f"smi:{data.smiles}  edge_index:\n{data.edge_index}  edge_attr:\n{data.edge_attr} ")

#out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch)# use our own x and edge_attr instead of data.x and data.edge_attr
#print(f"out:{out}")
#for data in data_loader:
#	x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
#	#print(f"before- data.x:{data.x.shape}, edge_attr:{data.edge_attr.shape}")
#	data.x = x
#	data.edge_attr = edge_attr
#	data.to(device)
#	out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch)# use our own x and edge_attr instead of data.x and data.edge_attr
#	print(f"out:{out}")
#


def train(data_loader, debug_mode):
	model.train()
	for data in data_loader:
		#print(f"smi:{data.smiles}")
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		#print(f"before- data.x:{data.x.shape}, edge_attr:{data.edge_attr.shape}")
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)
	
		#print(f"data.x:{data.x.shape}")
		#print(f"data.edge_attr:{data.edge_attr.shape}")
		out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch)# use our own x and edge_attr instead of data.x and data.edge_attr
		#print(f"out:{out},y:{data.y}")
		loss = criterion(out, data.y)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		if(debug_mode):
			out_list = out.cpu().detach().numpy()
			y_list = data.y.cpu().detach().numpy()
			#print(f"{len(out_list)}, {len(y_list)}")
			for i in range(len(out_list)): 
				print(f"{out_list[i][0]}, {y_list[i][0]}") # for making correlation plot

def test(data_loader, debug_mode):
	model.eval()

	squared_error_sum = 0 
	for data in data_loader:
		#data.to(device)
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		#x.to(device)
		#edge_attr.to(device)
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)	

		out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch) # use our own x and edge_attr instead of data.x and data.edge_attr
		pred = out
		#print(f"pred:{len(pred)},data.y:{len(data.y)}")
		t = sum(pow((pred - data.y),2)).cpu().detach().numpy()
		if(debug_mode):
			p = pred.cpu().detach().numpy()
			y = data.y.cpu().detach().numpy()
			#for debugging
#			print(f"pred:============")
#			for i in range(len(p)):
#				print(p[i][0])
#			print(f"y:============")
#			for i in range(len(y)):
#				print(y[i][0])
#			print(f"diff:============")
#			for i in range(len(y)):
#				print((p[i]-y[i])[0])
#			print(f"pow:============")
#			for i in range(len(y)):
#				print((pow(p[i]-y[i],2))[0])
#			print(f"sum=======")
#			print(t)


			# for plotting 
			out_list = out.cpu().detach().numpy()
			y_list = data.y.cpu().detach().numpy()
			#print(f"{len(out_list)}, {len(y_list)}")
			for i in range(len(out_list)): 
				print(f"{out_list[i][0]}, {y_list[i][0]}") # for making correlation plot
		squared_error_sum +=t

	num_samples = get_num_samples(data_loader)
	MSE = squared_error_sum / num_samples
	if(debug_mode):
		pass
		#print(f"squared_error_sum: {squared_error_sum}, len:{num_samples}, MSE:{MSE}")	
	return MSE[0]

def get_num_samples(data_loader):
	num_graph_in_last_batch = list(data_loader)[-1].num_graphs
	total = (len(data_loader)-1)* batch_size + num_graph_in_last_batch
	
	#print(f"len(data_loader):{len(data_loader)}, last batch:{num_graph_in_last_batch},  total:{total}")
	return total 

for epoch in range(num_epoches):
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.0007 * math.exp(-epoch/30 ))#, weight_decay = 5e-4)
	train(train_loader, False)#epoch==(num_epoches-1))
	train_MSE = test(train_loader, False)#  epoch==(num_epoches-1))
	test_MSE = test(test_loader,  epoch==(num_epoches-1))
	print(f"Epoch:{epoch:03d}, Train MSE:{train_MSE: .4f}, Test MSE:{test_MSE: .4f}")
