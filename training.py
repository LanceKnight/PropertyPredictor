import torch
from torch import topk
from torch.nn import BCELoss, CosineSimilarity
from torch.utils.data import BatchSampler

from torch_geometric.data import DataLoader
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Batch

import numpy as np
from statistics import mean
from itertools import compress


from molecule_processing import batch2attributes
from dataset_cv import SIDER


def BCELoss_no_NaN(out, target):
	target_no_NaN = target[~torch.isnan(target)]
	out = out[~torch.isnan(target)]
	target_no_NaN = target_no_NaN#target_no_NaN.detach() 
	loss = BCELoss()(out, target_no_NaN)
	if(loss is None):
		print(f"BCEloss:{loss}")
		print(f"out:\n{out}")
		print(f"target:\n{target}")
	return loss

#topk_matrix = None
#topk_indices = None
#bottomk_matrix = None
#bottomk_indices =None

#def calculate_top_bottomk_matrix(i, n):
#	global topk_matrix
#	global bottomk_matrix
#	global topk_indices
#	global bottomk_indices
#
#	if topk_matrix is None:
#		print('calculating topk/bottomk matrix...')
#		topk_matrix, topk_indices = torch.topk(SIDER.similarity_matrix, n)
#		bottomk_matrix, bottomk_indices = torch.topk(-SIDER.similarity_matrix, n)
#		print('done!')
#	pos_matrix = topk_matrix[i]
#	pos_indices = topk_indices[i]
#	neg_matrix = bottomk_matrix[i]
#	neg_indices = bottomk_indices[i]
#	return pos_matrix, pos_indices, neg_matrix, neg_indices

def get_topk(i,n):
	matrix, indices = torch.topk(SIDER.similarity_matrix.cpu()[i], n)
	#matrix, indices, _,_ = calculate_top_bottomk_matrix(i, n)
	return matrix, indices

def get_bottomk(i, n):
	matrix, indices = torch.topk(-SIDER.similarity_matrix.cpu()[i], n)
	#_, _, matrix, indices =  calculate_top_bottomk_matrix(i, n)	
	return -matrix, indices

def infoNCE(anchors, positives, negatives, device):
	anchors = anchors.view(positives.shape[0],1, positives.shape[2])
	anchors = anchors.expand(positives.shape)
	cos = CosineSimilarity(dim = 2)
	A = torch.mean(cos(anchors, positives))
	B = torch.mean(cos(anchors, negatives))
	

	#loss = -torch.log(torch.exp(A)/torch.exp(A+B))
	A_part = -torch.log(torch.tensor(1/2)*(A+torch.tensor(1)))
	B_part = -torch.log(torch.tensor(-1/2)*(B+torch.tensor(-1)))
	#print(f"A_part:{A_part} B_part:{B_part}")
	loss =   A_part + B_part
	#print(f"A:{A}, B:{B}, infoNCE loss:{loss}")
	return loss

def get_infoNCE(data, model, n, device):
	i = data.id
	positives = []
	negatives = []
	_, positive_indices = get_topk(i, n)
	_, negative_indices = get_bottomk(i, n)


	positive_sampler = BatchSampler(torch.flatten(positive_indices).tolist(), n, drop_last = False)
	#print(f"pos_sampler:{np.asarray(list(positive_sampler)).shape}")
	positive_dataloader = DataLoader(SIDER,batch_sampler = positive_sampler)
	#print('positive_dataloader done')
	del positive_sampler
	for i, pos_data in enumerate(positive_dataloader):
		#print(i)
		pos_data.to(device)
		_, pos_z=  model(pos_data.x.float(),pos_data.edge_index, pos_data.edge_attr, pos_data.smiles, pos_data.batch, False)
		positives.append(pos_z)
	#print('before stack')
	positives = torch.stack(positives)
	#print('after stack')
	
	negative_sampler = BatchSampler(torch.flatten(negative_indices).tolist(), n, drop_last = False)
	negative_dataloader = DataLoader(SIDER,batch_sampler = negative_sampler)
	for i,  neg_data in enumerate(negative_dataloader):
		#print(i)
		neg_data.to(device)
		_, neg_z = model(neg_data.x.float(),neg_data.edge_index, neg_data.edge_attr, neg_data.smiles, neg_data.batch, False)
		negatives.append(neg_z)
	negatives = torch.stack(negatives)
	#print(f"anchors1:{anchors}")
	  
	#negatives = [model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True) for data in dataset[negative_indices]]
	_,anchors_z = model( data.x.float(),data.edge_index, data.edge_attr, data.smiles, data.batch, False)
	return infoNCE(anchors_z, positives, negatives, device )

def get_unsupervised_loss(method=None, model = None, data = None, device=None, **kwargs):

	#print(data)
	loss = torch.tensor([0], device = device)
	if method == 'pi-model':
		edge_dropout_rate = kwargs['edge_dropout_rate']
		try:
			edge_index2, edge_attr2 =  dropout_adj(data.edge_index, data.edge_attr, p = edge_dropout_rate)

			out2, z2 = model(data.x.float(), edge_index2, edge_attr2, data.smiles, data.batch, False)# use our own x and edge_attr instead of data.x and data.edge_attr
			out2 = out2.view(len(data.y))

			out3, z3 = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, False)# use our own x and edge_attr instead of data.x and data.edge_attr
			out3 = out3.view(len(data.y))
			loss = torch.nn.MSELoss()(z2, z3)
		except Exception as e:
			print(f"exception:{e}")
			print(f"smi:{data.smiles} edge_index:{data.edge_index.shape} edge_attr:{data.edge_attr.shape}, p = {edge_dropout_rate}")
		#edge_index3, edge_attr3 =  dropout_adj(data.edge_index, data.edge_attr, p = edge_dropout_rate)
		#print(f"edge_index:{edge_index2.shape} edge_attr:{edge_attr2.shape}")
		#print(f"edge_index:{edge_index3.shape} edge_attr:{edge_attr3.shape}")
	elif method == 'infonce':
		num_pos_neg_samples =kwargs['num_pos_neg_samples'] 
		loss = get_infoNCE(data, model, num_pos_neg_samples, device)
	else:
		print('cannot get unsupervised loss!')

	return loss



def get_loss(method=None, data=None, model=None, predicted = None, y = None, device=None, unsupervised_weight=None, use_SSL = True, **kwargs):
	'''
	provide either (data, model) or (predicted, y), and return loss
	if using semi-supervised learning, return total_loss, supervised_loss, raw_unsupervised_loss
	otherwise, return supervised_loss
	'''
	mode = 0 # === 1 for (data, model), 2 for (predicted, y)
	if((data is None) or (model is None)):
		print('Error in getting loss: data and model should always be provided')
	else:
		if (predicted is not None) and (y is not None):
			mode = 2
		else:
			mode = 1

	if(mode ==1):# ===if provided (data, model) only
		out, z  = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True)# use our own x and edge_attr instead of data.x and data.edge_attr
		y = data.y
	elif(mode ==2):# ===if provided (predicted, y) as well
		out, z = predicted

	total_loss = torch.tensor(0)
	out = out.view(len(y))

	#===filter out data that contains only nan
	if( not torch.all( torch.isnan(data.y))):
		supervised_loss = BCELoss_no_NaN(out, y)
	else:
		#print(y)
		supervised_loss = torch.tensor(0)
	method= method.lower()
	methods = {'infonce', 'pi-model'}
	assert method in methods, 'unsupervised method does not exist!' 
	# === unsupervised learning
	if use_SSL == True:
		unsupervised_loss = get_unsupervised_loss(method = method,  model = model, data = data, device = device,**kwargs ) 
		total_loss = supervised_loss + unsupervised_weight * unsupervised_loss
		return total_loss, float(supervised_loss), float(unsupervised_loss)
	else:
		total_loss = supervised_loss
		return total_loss
		
	

def train(method =None, model=None, data_loader=None, target_col=None, unsupervised_weight=None, device=None, optimizer=None, use_SSL=True,  **kwargs):
	model.train()
	unsupervised_loss_lst = []
	supervised_loss_lst = []
	total_loss_lst = []
	for i,  data in enumerate(data_loader):
		# ===replace column y
		data.y = data.y[:,target_col]
		data.to(device)
		#print(f"smi:{data.smiles}\nx:\n{data.x}\n edge_index:\n{data.edge_index}\n edge_attr:{data.edge_attr}")
			



		# === filter out data that does not have y label (i.e. y label is Nan)
#		data_lst = data.to_data_list()#~torch.isnan(data.y)]
#		data_lst = list(compress(data_lst, list(~torch.isnan(data.y).cpu().numpy())))
#
#
#		if(len(data_lst)!=0): # === if there is labeled data in this batch, then get the supervised loss, otherwise the loss is 0
#
#			data = Batch.from_data_list(data_lst).to(device)
#			out, z  = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True)# use our own x and edge_attr instead of data.x and data.edge_attr
#			out = out.view(len(data.y))
#
#			supervised_loss = BCELoss()(out, data.y)
#		else:
#			supervised_loss = torch.tensor(0)
		
		if use_SSL:
			total_loss, supervised_loss, unsupervised_loss = get_loss(method=method,data = data, model = model, device = device, unsupervised_weight = unsupervised_weight, use_SSL=use_SSL, **kwargs)

			#if (supervised_loss<=0) or (torch.isnan(supervised_loss)):
			#	print(f"train_loss<=0: total_loss:{total_loss}, supervised_loss:{supervised_loss}, unsupervised_loss:{unsupervised_loss}  data:{data}")
			#	print(f"data.y:{data.y}")
		else:
			total_loss = get_loss(method=method,data = data, model = model, device = device, unsupervised_weight = unsupervised_weight, use_SSL=use_SSL, **kwargs)



		total_loss_lst.append(total_loss.item())
		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		total_loss = mean(total_loss_lst)
	return total_loss
