import torch
import torch.nn.functional as F
from torch.nn import Linear, Tanh, Softmax, Sigmoid, ReLU
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.data import DataLoader

import math
from statistics import mean
import numpy as np
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from molecule_processing import batch2attributes, num_node_features, num_edge_features
 

num_epoches = 100
inner_atom_dim = 512
batch_size = 64
hidden_activation = Softmax()#Tanh()
conv_depth = 5
discriminator_hidden_dim =128
discriminator_output_dim = 512
target_col = [x for x in range(0,6)]

print(f"target_col:{target_col}")
SIDER = MoleculeNet(root = "../data/raw/SIDER", name = "SIDER")
#print("data info:")
#print("============")
#print(f"num of data:{len(SIDER)}")

num_data = len(SIDER)
train_num = 1000#int(num_data * 0.8)
val_num = int(num_data * 0.0)
test_num = 100#num_data - train_num - val_num
print(f"train_num = {train_num}, val_num = {val_num}, test_num = {test_num}")

train_loader = DataLoader(SIDER[:train_num], batch_size = batch_size, shuffle = False)
validate_loader = DataLoader(SIDER[train_num:train_num+val_num], batch_size = batch_size, shuffle = False)
test_loader = DataLoader(SIDER[1427-test_num:1427], batch_size = test_num, shuffle = False)

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
		atom_fp_lst = []
		molecule_fp_lst = []
		for i in range(0, self.depth+1):
			atom_fp = Softmax(dim=1)(self.W_out(x))
			atom_fp_lst.append(atom_fp)	
			molecule_fp = global_add_pool(atom_fp, batch)
			molecule_fp_lst.append(molecule_fp)
			x = self.atom_bond_conv(x, edge_index, edge_attr, smiles, batch)

		overall_molecule_fp	= torch.stack(molecule_fp_lst, dim=0).sum(dim=0)	
		all_atom_fp = torch.cat(atom_fp_lst, dim=1) # a list of atom fingerprints. The fingerprints have fingerprints from all depth all concatenated together
		hidden = self.lin1(overall_molecule_fp)
		out = self.lin2(hidden)
		return Sigmoid()(out), all_atom_fp

class DiscriminateNet(torch.nn.Module):
	def __init__(self, vector_dim, hidden_dim, output_dim):
		super(DiscriminateNet, self).__init__()
		self.lin1 = Linear(vector_dim, hidden_dim)
		self.lin2 = Linear(hidden_dim, hidden_dim)
		self.lin3 = Linear(hidden_dim, output_dim)
	def forward(self, embedding):
		h1 = ReLU(self.lin1(embedding))
		h2 = self.lin2(h1)
		output = self.lin3(h2)
		return output
		

def softplus(z):
	return log(1+math.exp(z))

def discriminate(patch_embedding, graph_embedding):
	dis_net1 = DiscriminateNet(path_embedding.shape[1], discriminator_hidden_dim, discriminator_output_dim)
	dis_net2 = DiscriminateNet(graph_embedding.shape[1], discriminator_hidden_dim, discriminator_output_dim)
	patch_transform = dis_net1(patch_embedding)
	graph_transform = dis_net2(graph_embedding)

	discriminate_score = torch.dot(patch_transform1, graph_transform)
	return discriminate_score


def mutal_info(patch_embeddings, graph_embeddings, batch):
	"""
		node_graph_label: the patch-graph pair's label, that tells if a patch belongs to a graph
	"""
	postive_scores = []
	for i in range(graph_embeddings.shape[0]):
		graph_embedding = graph_embeddings[i]
		patch_embeddings_for_a_graph = patch_embeddings[idx for idx, graph_no in enumerate(batch) if graph_no = i ]
		positive_scores.append( -softplus(-discriminate(patch, graph)))
	negative_scores = softplus(discriminate(patch, graph)

	



#	for i, patch in enumerate(patch_embeddings):
#		for j,graph in enumerate(graph_embeddings):
#			if patch_graph_label[i][j] == 1:
#				positive_scores.append(-softplus(-discriminate(patch, graph)))
#
#			#negative_scores.append(softplus(discriminate(patch, graph))
		
	positive_score = mean(positive_scores)
	negative_score = mean(negative_scores)

	return positive_score - negative_score
		


is_cuda = torch.cuda.is_available()
#print(f"is_cuda:{is_cuda}")

device = torch.device('cuda' if is_cuda else 'cpu')
model = MyNet(num_node_features, num_edge_features, conv_depth).to(device)
#criterion = torch.nn.BCELoss()

def BCELoss_no_NaN(out, target):
	#print(f"out.shape:{out.shape}             target.shape:{target.shape}")
	target_no_NaN = torch.where(torch.isnan(target), out, target)
	target_no_NaN = target_no_NaN.detach() 
	#print(f"target_no_NaN:{target_no_NaN}")
	return torch.nn.BCELoss()(out, target_no_NaN)

def train(data_loader, debug_mode, target_col):
	model.train()
	for data in data_loader:
		print(f"smi:{len(data.smiles)}")
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		#print(f"before- data.x:{data.x.shape}, edge_attr:{data.edge_attr.shape}")
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)
	
		#print(f"data.x:{data.x.shape}")
		#print(f"data.edge_attr:{data.edge_attr.shape}")
		out, all_atom_fp  = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch)# use our own x and edge_attr instead of data.x and data.edge_attr
		print(f"node_fp_lst.shape {len(all_atom_fp)}, data.batch.shape {data.batch.shape}")
		graph_fp = out
		out = out.view(len(data.y[:,target_col]))
		#print(f"out.shape:{out.shape},           y.shape{data.y[:, target_col].shape}")
		#print(f"out:{out}\n y:\n{data.y[:,target_col]}")
		loss = BCELoss_no_NaN(out, data.y[:,target_col])
		#print(f"loss:{loss}")
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		if(debug_mode):
			out_list = out.cpu().detach().numpy()
			y_list = data.y.cpu().detach().numpy()
			#print(f"{len(out_list)}, {len(y_list)}")
			for i in range(len(out_list)): 
				print(f"{out_list[i][0]}, {y_list[i][0]}") # for making correlation plot

def test(data_loader, debug_mode, target_col):
	model.eval()

	auc_lst = []
	for data in data_loader:
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)	

		out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch) # use our own x and edge_attr instead of data.x and data.edge_attr

		#==========convert to numpy array
		out = out.view(len(out))	
		out = out.cpu().detach().numpy()
		#print(f"out:{out}")
		y = data.y[:,target_col]
		y = y.view(len(y)).cpu().detach().numpy()
		#print(f"y:{y}")
		#==========remove NaN
		out = out[~np.isnan(y)]
		y = y[~np.isnan(y)]

		#print(f"data.y.shape:{y}   out.shape:{out})")
		sc = roc_auc_score(y, out)
		auc_lst.append(sc)
		if(debug_mode):
			#p = pred.cpu().detach().numpy()
			#y = data.y.cpu().detach().numpy()
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
	if(debug_mode):
		pass
		#print(f"squared_error_sum: {squared_error_sum}, len:{num_samples}, MSE:{MSE}")	
	return mean(auc_lst)

def get_num_samples(data_loader):
	num_graph_in_last_batch = list(data_loader)[-1].num_graphs
	total = (len(data_loader)-1)* batch_size + num_graph_in_last_batch
	
	#print(f"len(data_loader):{len(data_loader)}, last batch:{num_graph_in_last_batch},  total:{total}")
	return total 

col_result = []
for col in target_col:
	print(f"col:{col}")
	test_sc = 0
	for epoch in tqdm(range(num_epoches)):
		optimizer = torch.optim.Adam(model.parameters(), lr = 0.0007 * math.exp(-epoch/30 ))#, weight_decay = 5e-4)
		train(train_loader, False, col)#epoch==(num_epoches-1))
		#train_sc = test(train_loader, False)#  epoch==(num_epoches-1))
		test_sc = test(test_loader, False, col)# epoch==(num_epoches-1))
		#print(f"Epoch:{epoch:03d}, Train AUC:{train_sc: .4f}, Test AUC:{test_sc: .4f}")
		if(epoch==num_epoches-1):
			print(f"Epoch:{epoch:03d}, Test AUC:{test_sc: .4f}")
	col_result.append(col_result)
#print(col_result)
