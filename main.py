# pi model with more unlabeled data
import torch
import torch.nn.functional as F
from torch.nn import Linear, Tanh, Softmax, Sigmoid, Dropout
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops

#import sys
import math
from statistics import mean
import numpy as np
from tqdm import tqdm
import pandas as pd
from clearml import Task
from argparse import ArgumentParser
from termcolor import cprint

from dataset_cv import get_loaders_with_idx, get_loaders, get_stats
from printing import tee_print, set_output_file, print_val_test_auc
from config_parser import get_config, set_config_file, get_config_dict
from training import train
from testing import test#, get_loss
from network import build_model
from param_grid_search import generate_param_sets, get_param_set, get_param_sets_length, record_result, save_file 


# load the configuration file
parser = ArgumentParser()
parser.add_argument('config', help="configration file")
args = parser.parse_args()

config_file = args.config
set_config_file(config_file)
task_name =get_config('cfg', 'task_name')

# set up ClearMl monitoring
task = Task.init(project_name='side_effect', task_name=task_name)
task.connect_configuration(configuration=config_file, name='configuration_file')
logger = task.get_logger()

# some configurations need to be load before param set
target_col = get_config('cfg','target_col')
use_SSL = bool(int(get_config('unsupervised', 'use_ssl')))

#if use_SSL == False:
#	num_extra_data = 0
#else:
#	num_extra_data = int(get_config('unsupervised','num_extra_data'))

output_file = get_config('file','output_file')
final_auc_file = get_config('file','auc_file')
auc_file_per_epoch = get_config('file', 'auc_file_per_epoch')

# set console output file for tee_print
set_output_file(output_file)

# generate param sets
model_config_dict = get_config_dict()
#model_config_dict['num_extra_data']=num_extra_data
task.connect(model_config_dict)

#print(model_config_dict)

param_sets = generate_param_sets(model_config_dict)

logger.report_table("param_set", table_plot=param_sets, iteration=0, series = '')

num_param_sets = get_param_sets_length()
print(f"there are {num_param_sets} param sets")
print(f"======")
for param_set_id in range(num_param_sets):
	param_set = get_param_set(param_set_id)
	print(param_set)
	col = int(param_set['target_col'])
	fold = int(param_set['fold'])

	inner_atom_dim = int(param_set['inner_atom_dim'])
	conv_depth = int(param_set['conv_depth'])
	sup_dropout_rate = float(param_set['sup_dropout_rate'])

	batch_size =  int(param_set['batch_size'])
	num_epochs =  int(param_set['num_epochs'])
	patience =   int(param_set['patience'])

	use_SSL =  bool(int(param_set['use_ssl']))
	cprint(f'use_SSL:{use_SSL}', 'red')
	method = param_set['method']
	cprint(f'method:{method}', 'red')
	rampup_length =  int(param_set['rampup_length'])
	unsup_dropout_rate =  float(param_set['unsup_dropout_rate'])
	w =  float(param_set['w'])
	edge_dropout_rate =  float(param_set['edge_dropout_rate'])
	num_pos_neg_samples = int(param_set['num_pos_neg_samples'])
	if use_SSL == True:
		num_extra_data =  int(param_set['num_extra_data'])
	else:
		num_extra_data = 0

	lr_init =  float(param_set['lr_init'])
	lr_base =  float(param_set['lr_base'])
	lr_exp_multiplier =  float(param_set[ 'lr_exp_multiplier'])


	print('------')


	# load data
	train_loader, val_loader, test_loader = get_loaders(num_extra_data, batch_size, col)
	#train_loader, val_loader, test_loader = get_loaders_with_idx(num_extra_data, batch_size, 1)
			
			
	is_cuda = torch.cuda.is_available()

	device = torch.device('cuda' if is_cuda else 'cpu')

	def rampup(epoch):
		 if epoch < rampup_length:
			  p = max(0.0, float(epoch)) / float(rampup_length)
			  p = 1.0 - p
			  return math.exp(-p*p*5.0)
		 else:
			  return 1.0



#	def get_num_samples(data_loader):
#		num_graph_in_last_batch = list(data_loader)[-1].num_graphs
#		total = (len(data_loader)-1)* batch_size + num_graph_in_last_batch
#		
#		#print(f"len(data_loader):{len(data_loader)}, last batch:{num_graph_in_last_batch},  total:{total}")
#		return total 

	train_auc_per_epoch = ['train']
	val_auc_per_epoch = ['val']
	test_auc_per_epoch = ['test']
	lrs = []



	num_train, num_labels, num_unlabeled = get_stats(col)
	test_sc = 0
	args = {'inner_atom_dim': inner_atom_dim, 'sup_dropout_rate': sup_dropout_rate, 'conv_depth':conv_depth, 'unsup_dropout_rate': unsup_dropout_rate}
	model = build_model(device, **args)
	optimizer = torch.optim.Adam(model.parameters(), lr = lr_init)#, weight_decay = 5e-4)
	lambda1 = lambda epoch: lr_base ** (lr_exp_multiplier*epoch)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
	previous_val_sc = 999
	patience_count = 0
	
	unsupervised_param = {'edge_dropout_rate': edge_dropout_rate,
								 'num_pos_neg_samples' : num_pos_neg_samples
								}

	for epoch in tqdm(range(num_epochs)):	
		# === setting up
		lr = optimizer.param_groups[0]["lr"]
		rampup_val = rampup(epoch)
		unsupervised_weight = rampup_val * w
		# === training
		train_loss_in_progress = train(method=method, model = model, data_loader = train_loader,  target_col = col, unsupervised_weight = unsupervised_weight, device = device,  optimizer=optimizer, use_SSL = use_SSL, **unsupervised_param)
		scheduler.step()
		# ===reporting
		if(use_SSL == True):	# === report unsupervised and supervised loss if using semi-supervised learning
		
			print('-----training')	
			train_sc, train_loss, train_supervised_loss, train_unsupervised_loss= map(lambda x: round(x,4),test(model= model, data_loader = train_loader,  target_col =col, device = device, method = method, unsupervised_weight = unsupervised_weight, use_SSL = use_SSL, **unsupervised_param))
			print('----validation')
			val_sc, val_loss,val_supervised_loss, val_unsupervised_loss=   map(lambda x: round(x,4),test(model= model, data_loader = val_loader,  target_col =col, device = device, method = method, unsupervised_weight = unsupervised_weight, use_SSL = use_SSL, **unsupervised_param))
			print('---testing')
			test_sc, test_loss, test_supervised_loss, test_unsupervised_loss =   map(lambda x: round(x,4),test(model= model, data_loader = test_loader,  target_col =col, device = device, method = method, unsupervised_weight = unsupervised_weight, use_SSL = use_SSL, **unsupervised_param))
	

			logger.report_scalar(title=f'supervised loss for param set {param_set_id}', series = 'train supervised loss', value =train_supervised_loss,  iteration = epoch)
			logger.report_scalar(title=f'supervised loss for param set {param_set_id}', series = 'validation supervised loss', value =val_supervised_loss,  iteration = epoch)
			logger.report_scalar(title=f'supervised loss for param set {param_set_id}', series = 'test supervised loss', value =test_supervised_loss,  iteration = epoch)

			logger.report_scalar(title=f'raw unsupervised loss for param set {param_set_id}', series = 'train unsupervised loss', value =train_unsupervised_loss,  iteration = epoch)
			logger.report_scalar(title=f'raw unsupervised loss for param set {param_set_id}', series = 'validation unsupervised loss', value =val_unsupervised_loss,  iteration = epoch)
			logger.report_scalar(title=f'raw unsupervised loss for param set {param_set_id}', series = 'test unsupervised loss', value =test_unsupervised_loss,  iteration = epoch)
		else: # === only report supervised loss if not using semi-supervised learning
			train_sc, train_loss2= map(lambda x: round(x,4),test(model= model, data_loader = train_loader,  target_col =col, device = device, method = method, unsupervised_weight = unsupervised_weight, use_SSL = use_SSL, **unsupervised_param))
			val_sc, val_loss =   map(lambda x: round(x,4),test(model= model, data_loader = val_loader,  target_col =col, device = device, method = method, unsupervised_weight = unsupervised_weight, use_SSL = use_SSL, **unsupervised_param))
			test_sc, test_loss =   map(lambda x: round(x,4),test(model= model, data_loader = test_loader,  target_col =col, device = device, method = method, unsupervised_weight = unsupervised_weight, use_SSL = use_SSL, **unsupervised_param))


		logger.report_scalar(title=f'total loss for param set {param_set_id}', series = 'train loss', value =train_loss,  iteration = epoch)
		#logger.report_scalar(title=f'total loss for param set {param_set_id}', series = 'train loss2', value =train_loss2,  iteration = epoch)
		logger.report_scalar(title=f'total loss for param set {param_set_id}', series = 'validation loss', value =val_loss,  iteration = epoch)
		logger.report_scalar(title=f'total loss for param set {param_set_id}', series = 'test loss', value =test_loss,  iteration = epoch)

		logger.report_scalar(title=f'learning rate for param set {param_set_id}', series = 'learning rate', value =lr,  iteration = epoch)
		logger.report_scalar(title=f'ROC-AUC for param set { param_set_id}', series = 'training', value =train_sc,  iteration = epoch)
		logger.report_scalar(title=f'ROC-AUC for param set { param_set_id}', series = 'validation', value =val_sc,  iteration = epoch)
		logger.report_scalar(title=f'ROC-AUC for param set { param_set_id}', series = 'testing', value =test_sc,  iteration = epoch)

		# ===early stop
		if val_sc - previous_val_sc <0.0001:
			patience_count +=1
			if(patience_count == patience):
				print(f"consecutive {patience} epochs without validation set improvement. Break early at epoch {epoch}")
				break
		else:
			patience_count = 0			

		if((epoch%1 == 0)):
			pass
			#print(f"Epoch:{epoch:03d}, train AUC: {train_sc: .4f}   val AUC: {val_sc: .4f}  test_AUC:{test_sc:.4f}")
		previous_val_sc = val_sc




	tee_print(f"col:{col}, extra_unlabeled:{num_extra_data}, w:{w}     train_sc:{train_sc:.4f} val_sc:{val_sc:.4f} test AUC: {test_sc:.4f}")
	record_result(param_set_id, 'train_auc', train_sc)
	record_result(param_set_id, 'val_auc', val_sc)
	record_result( param_set_id, 'test_auc', test_sc)
	
	print(f"======")
	logger.report_histogram(title = f"param set comparison - param set {param_set_id}", series = 'training',values=train_sc, iteration = 0, xaxis = 'param set', yaxis ='AUC', mode = 'group')
	logger.report_histogram(title = f"param set comparison - param set {param_set_id}", series = 'validation', values=val_sc, iteration = 0, xaxis = 'param set', yaxis ='AUC', mode = 'group')
	logger.report_histogram(title = f"param set comparison - param set {param_set_id}", series = 'testing', values=test_sc, iteration = 0, xaxis = 'param set', yaxis ='AUC', mode = 'group')

print_val_test_auc(train_auc_per_epoch, val_auc_per_epoch, test_auc_per_epoch, auc_file_per_epoch)
save_file()
