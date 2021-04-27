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
from tqdm import tqdm
import pandas as pd

from dataset import get_loaders, get_stats
from printing import tee_print, set_output_file, print_val_test_auc
from config_parser import get_config, set_config_file, get_config_dict
from training import train
from testing import test
from network1 import build_model


config_file = sys.argv[1]
set_config_file(config_file)

target_col = get_config('cfg','target_col')

inner_atom_dim = int(get_config('architecture','inner_atom_dim'))
conv_depth = int(get_config('architecture','conv_depth'))
sup_dropout_rate = float(get_config('architecture','sup_dropout_rate'))

batch_size = int(get_config('training','batch_size'))
num_epochs = int(get_config('training','num_epochs'))
patience = int(get_config('training','patience'))

use_SSL = bool(int(get_config('unsupervised', 'use_ssl')))
unsup_dropout_rate = float(get_config('unsupervised','unsup_dropout_rate'))
w = get_config('unsupervised','w')
edge_dropout_rate = float(get_config('unsupervised','edge_dropout_rate'))
rampup_length = int(get_config('unsupervised','rampup_length'))
if use_SSL == False:
	num_extra_data = 0
else:
	num_extra_data = int(get_config('unsupervised','num_extra_data'))

lr_init_lst = get_config('lr','lr_init')
lr_base_lst = get_config('lr', 'lr_base')
lr_exp_multiplier_lst = get_config('lr', 'lr_exp_multiplier')

output_file = get_config('file','output_file')
final_auc_file = get_config('file','auc_file')
auc_file_per_epoch = get_config('file', 'auc_file_per_epoch')



model_config_dict = get_config_dict()
model_config_dict['num_extra_data']=num_extra_data
print(model_config_dict)

set_output_file(output_file)

tee_print(f"target_col:{target_col} use_ssl:{use_SSL}")

train_loader, val_loader, test_loader = get_loaders(num_extra_data, batch_size)
		
		
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

train_auc = ['train']
val_auc = ['val']
test_auc = ['test']
train_auc_per_epoch = ['train']
val_auc_per_epoch = ['val']
test_auc_per_epoch = ['test']
lrs = []

#for edge_dropout_rate in edge_dropout_rate_lst:
	#edge_dropout_rate = float(edge_dropout_rate)
for lr_init in lr_init_lst:	
	train_auc.append(f"lr_init:{lr_init}")
	val_auc.append(f"lr_init:{lr_init}")
	test_auc.append(f"lr_init:{lr_init}")
	for lr_base in lr_base_lst:
		train_auc.append(f"lr_base:{lr_base}")
		val_auc.append(f"lr_base:{lr_base}")
		test_auc.append(f"lr_base:{lr_base}")
		for lr_exp_multiplier in lr_exp_multiplier_lst:
			train_auc.append(f"lr_exp_multiplier:{lr_exp_multiplier}")
			val_auc.append(f"lr_exp_multiplier:{lr_exp_multiplier}")
			test_auc.append(f"lr_exp_multiplier:{lr_exp_multiplier}")
			for ini_scaled_unsupervised_weight in w:
				tee_print("\n")		
				train_auc.append(ini_scaled_unsupervised_weight)
				val_auc.append(ini_scaled_unsupervised_weight)
				test_auc.append(ini_scaled_unsupervised_weight)

				train_auc_per_epoch.append(ini_scaled_unsupervised_weight)
				val_auc_per_epoch.append(ini_scaled_unsupervised_weight)
				test_auc_per_epoch.append(ini_scaled_unsupervised_weight)
				for col in target_col:
					num_train, num_labels, num_unlabeled = get_stats(col)
					#scaled_unsupervised_weight = ini_scaled_unsupervised_weight * float(num_labels) / float(num_train)
					test_sc = 0
					#model = MyNet(num_node_features, num_edge_features, conv_depth).to(device)#Get_Net(num_node_features, num_edge_features, conv_depth, inner_atom_dim, dropout_rate).to(device)
					args1 = get_config('unsupervised')
					args2 = get_config('architecture')
					model = build_model(device, **args1, **args2)
					optimizer = torch.optim.Adam(model.parameters(), lr = lr_init)#, weight_decay = 5e-4)
					lambda1 = lambda epoch: lr_base ** (lr_exp_multiplier*epoch)
					scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
					previous_val_sc = 999
					patience_count = 0
					train_auc.append(col)
					val_auc.append(col)
					test_auc.append(col)

					train_auc_per_epoch.append(col)
					val_auc_per_epoch.append(col)
					test_auc_per_epoch.append(col)
					for epoch in tqdm(range(num_epochs)):
						
						lrs.append(optimizer.param_groups[0]["lr"])
						rampup_val = rampup(epoch)
						unsupervised_weight = rampup_val * ini_scaled_unsupervised_weight
						train(model, train_loader,  col, unsupervised_weight, device, optimizer, use_SSL = use_SSL, debug_mode = False, edge_dropout_rate = edge_dropout_rate)#epoch==(num_epoches-1))
						scheduler.step()
						#train_sc = test(train_loader, False)#  epoch==(num_epoches-1))
						#print(f"Epoch:{epoch:03d}, Train AUC:{train_sc: .4f}, Test AUC:{test_sc: .4f}")
						#print(f"Epoch:{epoch:03d}, Test AUC:{test_sc: .4f}")
						train_sc = round(test(model, train_loader, False, col, device),4)
						val_sc = round(test(model, val_loader, False, col, device),4)
						test_sc = round(test(model, test_loader, False, col, device),4)
						train_auc_per_epoch.append(train_sc)
						val_auc_per_epoch.append(val_sc)
						test_auc_per_epoch.append(test_sc)
						if val_sc - previous_val_sc <0.0001:
							patience_count +=1
							if(patience_count == patience):
								print(f"consecutive {patience} epochs without validation set improvement. Break early at epoch {epoch}")
								break
						else:
							patience_count = 0			

						if((epoch%1 == 0)):
							print(f"Epoch:{epoch:03d}, train AUC: {train_sc: .4f}   val AUC: {val_sc: .4f}  test_AUC:{test_sc:.4f}")
						previous_val_sc = val_sc
					print(f"lrs:{lrs}")
					tee_print(f"col:{col}, extra_unlabeled:{num_extra_data}, w:{ini_scaled_unsupervised_weight}     train_sc:{train_sc:.4f} val_sc:{val_sc:.4f} test AUC: {test_sc:.4f}")
					train_auc.append(train_sc)
					val_auc.append(val_sc)
					test_auc.append(test_sc)
				
				
	
print_val_test_auc(train_auc, val_auc, test_auc,  final_auc_file)
print_val_test_auc(train_auc_per_epoch, val_auc_per_epoch, test_auc_per_epoch, auc_file_per_epoch)
