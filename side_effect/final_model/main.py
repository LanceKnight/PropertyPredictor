# pi model with more unlabeled data
import torch
import torch.nn.functional as F
from torch.nn import Linear, Tanh, Softmax, Sigmoid, Dropout
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops

import sys
import math
from statistics import mean
import numpy as np
#from rdkit.Chem import MolFromSmiles
#import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
#import rdkit.Chem.EState as EState
#import rdkit.Chem.rdPartialCharges as rdPartialCharges
#from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import pandas as pd

from molecule_processing import num_node_features, num_edge_features#, batch2attributes
from dataset import get_loaders, get_stats
from printing import tee_print, set_output_file, print_val_test_auc
from config_parser import get_config, set_config_file
from training import train
from testing import test


config_file = sys.argv[1]
set_config_file(config_file)

p = int(get_config('cfg','p'))
rampup_length = int(get_config('cfg','rampup_length'))
target_col = get_config('cfg','target_col')
num_extra_data = int(get_config('cfg','num_extra_data'))

inner_atom_dim = int(get_config('architecture','inner_atom_dim'))
#hidden_activation = get_config('architecture','hidden_activation')
conv_depth = int(get_config('architecture','conv_depth'))
sup_dropout_rate = float(get_config('architecture','sup_dropout_rate'))

batch_size = int(get_config('training','batch_size'))
num_epochs = int(get_config('training','num_epochs'))

unsup_dropout_rate = float(get_config('unsupervised','unsup_dropout_rate'))
w = get_config('unsupervised','w')
use_SSL = bool(int(get_config('unsupervised', 'use_ssl')))

lr_ini = get_config('lr','lr_init')
lr_base = get_config('lr', 'lr_base')
lr_exp_multiplier = get_config('lr', 'lr_exp_multiplier')

output_file = get_config('file','output_file')
final_auc_file = get_config('file','auc_file')
auc_file_per_epoch = get_config('file', 'auc_file_per_epoch')


set_output_file(output_file)

tee_print(f"target_col:{target_col} use_ssl:{use_SSL}")

train_loader, val_loader, test_loader = get_loaders(num_extra_data, batch_size)
		
class AtomBondConv(MessagePassing):
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
	def __init__(self, num_node_features, num_edge_features, depth):
		super(MyNet, self).__init__()
		self.atom_bond_conv = AtomBondConv(num_node_features, num_edge_features)
		self.W_out = Linear(num_node_features, inner_atom_dim)
		self.lin1 = Linear(inner_atom_dim, 50)
		self.lin2 = Linear(50, 1)
		self.depth = depth
		self.unsup_dropout = torch.nn.Dropout(unsup_dropout_rate)
		self.sup_dropout = torch.nn.Dropout(sup_dropout_rate)
	def forward(self, x, edge_index, edge_attr, smiles, batch, is_supervised):
		if is_supervised:
			dropout = self.sup_dropout
		else:
			dropout = self.unsup_dropout
		molecule_fp_lst = []
		for i in range(0, self.depth+1):
			atom_fp = Softmax(dim=1)(self.W_out(x))	
			molecule_fp = global_add_pool(atom_fp, batch)
			molecule_fp_lst.append(molecule_fp)
			x = self.atom_bond_conv(x, edge_index, edge_attr, smiles, batch, is_supervised)

		overall_molecule_fp	= torch.stack(molecule_fp_lst, dim=0).sum(dim=0)	
		hidden = dropout(self.lin1(overall_molecule_fp))
		out = dropout(self.lin2(hidden))
		return Sigmoid()(out)
		
is_cuda = torch.cuda.is_available()
#print(f"is_cuda:{is_cuda}")

device = torch.device('cuda' if is_cuda else 'cpu')
#criterion = torch.nn.BCELoss()

def rampup(epoch):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0




def get_num_samples(data_loader):
	num_graph_in_last_batch = list(data_loader)[-1].num_graphs
	total = (len(data_loader)-1)* batch_size + num_graph_in_last_batch
	
	#print(f"len(data_loader):{len(data_loader)}, last batch:{num_graph_in_last_batch},  total:{total}")
	return total 


val_auc = ['val']
test_auc = ['test']
val_auc_per_epoch = ['val']
test_auc_per_epoch = ['test']
lrs = []
for ini_scaled_unsupervised_weight in w:
	tee_print("\n")		
	val_auc.append(ini_scaled_unsupervised_weight)
	test_auc.append(ini_scaled_unsupervised_weight)

	val_auc_per_epoch.append(ini_scaled_unsupervised_weight)
	test_auc_per_epoch.append(ini_scaled_unsupervised_weight)
	for col in target_col:
		num_train, num_labels, num_unlabeled = get_stats(col)
		scaled_unsupervised_weight = ini_scaled_unsupervised_weight * float(num_labels) / float(num_train)
		test_sc = 0
		model = MyNet(num_node_features, num_edge_features, conv_depth).to(device)#Get_Net(num_node_features, num_edge_features, conv_depth, inner_atom_dim, dropout_rate).to(device)

		optimizer = torch.optim.Adam(model.parameters(), lr = 0.007)#, weight_decay = 5e-4)
		lambda1 = lambda epoch:math.exp(-epoch/30)
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
		previous_val_sc = 999
		patience_count = 0
		val_auc_per_epoch.append(col)
		test_auc_per_epoch.append(col)
		for epoch in tqdm(range(num_epochs)):
			
			lrs.append(optimizer.param_groups[0]["lr"])
			rampup_val = rampup(epoch)
			unsupervised_weight = rampup_val * scaled_unsupervised_weight
			train(model, train_loader,  col, unsupervised_weight, device, optimizer, use_SSL = use_SSL, debug_mode = False)#epoch==(num_epoches-1))
			scheduler.step()
			#train_sc = test(train_loader, False)#  epoch==(num_epoches-1))
			#print(f"Epoch:{epoch:03d}, Train AUC:{train_sc: .4f}, Test AUC:{test_sc: .4f}")
			#print(f"Epoch:{epoch:03d}, Test AUC:{test_sc: .4f}")
			val_sc = test(model, val_loader, False, col, device)
			test_sc = test(model, test_loader, False, col, device)# epoch==(num_epoches-1))
			val_auc_per_epoch.append(val_sc)
			test_auc_per_epoch.append(test_sc)
			if val_sc > previous_val_sc:
				patience_count +=1
				if(patience_count == p):
					print(f"consecutive {p} epochs without validation set improvement. Break early at epoch {epoch}")
					break
			else:
				patience_count = 0			

			if((epoch%1 == 0)):
				print(f"Epoch:{epoch:03d}, val AUC: {val_sc: .4f}  test_AUC:{test_sc:.4f}")
			previous_val_sc = val_sc
		print(f"lrs:{lrs}")
		tee_print(f"col:{col}, extra_unlabeled:{num_extra_data}, w:{ini_scaled_unsupervised_weight}  val_sc:{val_sc:.4f}             test AUC: {test_sc:.4f}")
		val_auc.append(val_sc)
		test_auc.append(test_sc)
	
	
	
print_val_test_auc(val_auc, test_auc,  final_auc_file)
print_val_test_auc(val_auc_per_epoch, test_auc_per_epoch, auc_file_per_epoch)

