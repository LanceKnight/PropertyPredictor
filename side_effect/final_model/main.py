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
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import pandas as pd

from molecule_processing import batch2attributes, num_node_features, num_edge_features
from dataset import Get_Loaders, Get_Stats
from printing import tee_print, set_output_file, print_val_test_auc
from config_parser import get_config, set_config_file


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

train_loader, val_loader, test_loader = Get_Loaders(num_extra_data, batch_size)
		
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

def BCELoss_no_NaN(out, target):
	#print(f"out.shape:{out.shape}             target.shape:{target.shape}")
	#target_no_NaN = torch.where(torch.isnan(target), out, target)
	target_no_NaN = target[~torch.isnan(target)]
	out = out[~torch.isnan(target)]
	target_no_NaN = target_no_NaN.detach() 
	
	#print(f"target_no_NaN:{target_no_NaN}")
	return torch.nn.BCELoss()(out, target_no_NaN)


def train(data_loader, debug_mode, target_col, unsupervised_weight):
	model.train()
	u_loss_lst = []
	s_loss_lst = []
	t_loss_lst = []
	for i,  data in enumerate(data_loader):
		#print(f"i:{i}, smi:{data.smiles}")
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		#print(f"before- data.x:{data.x.shape}, edge_attr:{data.edge_attr.shape}")
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)
	

		#print(f"data.x:{data.x.shape}")
		#print(f"data.edge_attr:{data.edge_attr.shape}")
		out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True)# use our own x and edge_attr instead of data.x and data.edge_attr
		out = out.view(len(data.y[:,target_col]))

		#print(f"out.shape:{out.shape},           y.shape{data.y[:, target_col].shape}")
		#print(f"out:{out}\n y:\n{data.y[:,target_col]}")
		loss = BCELoss_no_NaN(out, data.y[:,target_col])
		#print(f"out:\n{out}, out2:\n{out2} ")
		if use_SSL == True:
			out2 = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, False)# use our own x and edge_attr instead of data.x and data.edge_attr
			out2 = out2.view(len(data.y[:,target_col]))

			out3 = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, False)# use our own x and edge_attr instead of data.x and data.edge_attr
			out3 = out3.view(len(data.y[:,target_col]))
			unsupervised_loss = torch.nn.MSELoss()(out3, out2)
			total_loss = loss + unsupervised_weight * unsupervised_loss
			u_loss_lst.append(unsupervised_weight*unsupervised_loss.item())
			s_loss_lst.append(loss.item())
			t_loss_lst.append(total_loss.item())
			print(f"u_loss:{unsupervised_loss:.4f}  unsupervised_weight:{unsupervised_weight}    product:{unsupervised_weight* unsupervised_loss:.4f} loss:{loss:.4f} total loss:{total_loss:.4f}")
		else:
			total_loss = loss
		#print(f"loss:{loss}")
		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		if(debug_mode):
			out_list = out.cpu().detach().numpy()
			y_list = data.y.cpu().detach().numpy()
			#print(f"{len(out_list)}, {len(y_list)}")
			for i in range(len(out_list)): 
				print(f"{out_list[i][0]}, {y_list[i][0]}") # for making correlation plot
		
	u_loss = mean(u_loss_lst)
	s_loss = mean(s_loss_lst)
	t_loss = mean(t_loss_lst)
	print(f"u_loss:{u_loss:.4f}    s_loss:{s_loss:.4f}     t_loss:{t_loss:.4f}")


def roc_auc_score_one_class_compatible(y_true, y_predict):
	if len(set(y_true)) == 1:
		pass
	else:
		return roc_auc_score(y_true, y_predict)

def pr_auc(y_true, y_predict):
	precision, recall, _ = precision_recall_curve(y_true, y_predict)
	auc_score = auc(recall, precision)	
	return auc_score

def test(data_loader, debug_mode, target_col):
	model.eval()

	auc_lst = []
	for data in data_loader:
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)	

		out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True) # use our own x and edge_attr instead of data.x and data.edge_attr

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
		sc = roc_auc_score_one_class_compatible(y, out)
		#sc = pr_auc(y, out)
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
		num_train, num_labels, num_unlabeled = Get_Stats(col)
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
			train(train_loader, False, col, unsupervised_weight)#epoch==(num_epoches-1))
			scheduler.step()
			#train_sc = test(train_loader, False)#  epoch==(num_epoches-1))
			#print(f"Epoch:{epoch:03d}, Train AUC:{train_sc: .4f}, Test AUC:{test_sc: .4f}")
			#print(f"Epoch:{epoch:03d}, Test AUC:{test_sc: .4f}")
			val_sc = test(val_loader, False, col)
			test_sc = test(test_loader, False, col)# epoch==(num_epoches-1))
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

