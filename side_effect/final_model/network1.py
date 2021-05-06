import torch
from torch.nn import Linear, Tanh, Softmax, Sigmoid, Dropout, ReLU
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops


from molecule_processing import num_node_features, num_edge_features#, batch2attributes


class AtomBondConv(MessagePassing):
	global sup_dropout_rate
	global inner_atom_dim
	global unsup_dropout_rate
	def __init__(self, x_dim, edge_attr_dim):
		super(AtomBondConv, self).__init__(aggr = 'add')
		self.W_in = Linear(x_dim + edge_attr_dim, x_dim)
		#self.dropout = torch.nn.Dropout(dropout_rate)
		self.unsup_dropout = torch.nn.Dropout(unsup_dropout_rate)
		self.sup_dropout = torch.nn.Dropout(sup_dropout_rate)
	def forward(self, x, edge_index, edge_attr, smiles, batch, is_supervised ):		
		if is_supervised:
			dropout = self.sup_dropout
		else:
			dropout = self.unsup_dropout
		edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
		x = self.propagate(edge_index, x = x, edge_attr = edge_attr)
		x = dropout(self.W_in(x))
		return x

	def message(self, x, x_j, edge_attr):
		zero_tensor = torch.zeros(x.shape[0], edge_attr.shape[1], device = x_j.device) #create zero_tensor to pad the sec_col
		sec_col = torch.cat((edge_attr, zero_tensor), dim = 0) # create the second column. The first column has x_j, which is of shape (num_edge + num_node, num_node_feature), the second column has shape of (num_edge, edge_attr), padded with zero_tensor
		neighbor_atom_bond_feature = torch.cat((x_j, sec_col), dim = 1)
		return neighbor_atom_bond_feature


class MyNet(torch.nn.Module):
	global sup_dropout_rate
	global unsup_dropout_rate
	def __init__(self, num_node_features, num_edge_features, depth):
		super(MyNet, self).__init__()
		self.atom_bond_conv = AtomBondConv(num_node_features, num_edge_features)
		self.W_out = Linear(num_node_features, inner_atom_dim)
		self.predict_lin1 = Linear(inner_atom_dim, 50)
		self.predict_lin2 = Linear(50, 1)
		self.project_head_lin1 = Linear(inner_atom_dim, 128)
		self.project_head_lin2 = Linear(128, 1)
		self.depth = depth
	def forward(self, x, edge_index, edge_attr, smiles, batch, is_supervised):
		molecule_fp_lst = []

		atom_fp = Softmax(dim=1)(self.W_out(x))	
		molecule_fp = global_add_pool(atom_fp, batch)
		molecule_fp_lst.append(molecule_fp)
		for i in range(0, self.depth):
			atom_fp = Softmax(dim=1)(self.W_out(x))	
			molecule_fp = global_add_pool(atom_fp, batch)
			molecule_fp_lst.append(molecule_fp)
			x = self.atom_bond_conv(x, edge_index, edge_attr, smiles, batch, is_supervised)

		z = self.project_head_lin2(ReLU()(self.project_head_lin1(molecule_fp)))
		
		#overall_molecule_fp	= torch.stack(molecule_fp_lst, dim=0).sum(dim=0)	

		#adaptation layer
		hidden = self.predict_lin1(molecule_fp)
		out = self.predict_lin2(hidden)
		return Sigmoid()(out), z 

inner_atom_dim =0
sup_dropout_rate = 0
unsup_dropout_rate = 0
def build_model(device = '', **kwargs):
	global inner_atom_dim
	global sup_dropout_dim
	global unsup_dropout_rate
	#print(f"kwargs len:{len(kwargs)}")
	#for k in kwargs:
	#	print( k)
	
	inner_atom_dim =int(kwargs['inner_atom_dim'])
	sup_dropout_rate= float(kwargs['sup_dropout_rate'])
	conv_depth = int(kwargs['conv_depth'])
	unsup_dropout_rate = float(kwargs['unsup_dropout_rate'])

	return MyNet(num_node_features, num_edge_features, conv_depth).to(device)
