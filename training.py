import torch
from torch import topk
from torch.nn import BCELoss, CosineSimilarity
from torch.utils.data import BatchSampler

from torch_geometric.data import DataLoader
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Batch

from statistics import mean
from itertools import compress


from molecule_processing import batch2attributes
from dataset_cv import SIDER


def BCELoss_no_NaN(out, target):
	#print(f"out.shape:{out.shape}             target.shape:{target.shape}")
	#target_no_NaN = torch.where(torch.isnan(target), out, target)
	target_no_NaN = target[~torch.isnan(target)]
	out = out[~torch.isnan(target)]
	target_no_NaN = target_no_NaN.detach() 
	#print(f"target_no_NaN:{target_no_NaN}")
	return BCELoss()(out, target_no_NaN)

def get_unsupervised_loss(method=None, **kwargs):
	loss = torch.tensor([0], device = kwargs['device'])
	if method == 'pi_model':
		model = kwargs['model']
		data = kwargs['data']
		edge_dropout_rate = kwargs['edge_dropout_rate']
		target_col = kwargs['target_col']
		try:
			#print("hereaaaaaaaaaaaaaaaaaaaaaaaaaaa")
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
	elif method == 'edge_dropout':
		pass	
	else:
		print('cannot get unsupervised loss!')

	return loss

#def get_topk(i,n):
#	matrix, indices = topk(SIDER.similarity_matrix[i], n)
#	return matrix, indices
#
#def get_bottomk(i, n):
#	matrix, indices = topk(-SIDER.similarity_matrix[i], n)
#	return -matrix, indices
#
#def infoNCE(anchor, positives, negatives):
#	A = torch.tensor(anchor.size)
#	B = torch.tensor(anchor.size)
#	for positive in positives:
#		A = A+CosineSimilarity()(anchor, positive)
#	for negative in negatives:
#		B = B+CosineSimilarity()(anchor, nagative)
#
#	loss = A/(A+B)
#
#	return loss
#
#def get_infoNCE(anchors, model, n):
#	for i in range(anchors.shape[0]):
#		anchor = anchors[i]
#		_, positive_indices = get_topk(i,n)
#		_, negative_indices = get_bottomk(i, n)
#		positives = [model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True) for data in SIDER[positive_indices]]
#		negatives = [model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True) for data in SIDER[negative_indices]]
#		return infoNCE(anchor, positives, negatives )

def get_topk(i,n, device):
	matrix, indices = torch.topk(SIDER.similarity_matrix.cpu()[i], n)
	
	return matrix, indices

def get_bottomk(i, n, device):
	matrix, indices = torch.topk(-SIDER.similarity_matrix.cpu()[i], n)
	return -matrix, indices

def infoNCE(anchors, positives, negatives, device):
	A_list = []
	B_list = []
	#     print(f"positives_length:{len(positives)}")
	#     print(f"negative_length:{len(negatives)}")
	for i in range(anchors.shape[0]):
		positive = positives[i]
		#print(f"anchors:{anchors[i].view(1, anchors.shape[1]).shape}, positive:{positive.shape}")
		A_list.append(torch.mean(CosineSimilarity()(anchors[i].view(1,anchors.shape[1]), positive)))
		negative = negatives[i]
		B_list.append(torch.mean(CosineSimilarity()(anchors[i].view(1,anchors.shape[1]), negative)))

	A = torch.mean(torch.stack(A_list))
	B = torch.mean(torch.stack(B_list))
	#print(f"A:{A}, B:{B}")
	loss = -torch.log(torch.exp(A)/torch.exp(A+B))
	return loss

def get_infoNCE(data, model, n, device):
	#for i in range(anchors.shape[0]):
	i = data.id
	positives = []
	negatives = []
	_, positive_indices = get_topk(i, n, device)
	_, negative_indices = get_bottomk(i, n, device)
	#print(positive_indices)
	positive_sampler = BatchSampler(torch.flatten(positive_indices).tolist(), n, drop_last = False)
	positive_dataloader = DataLoader(SIDER,batch_sampler = positive_sampler)
	for pos_data in positive_dataloader:
		pos_data.to(device)
		_, pos_z=  model(pos_data.x.float(),pos_data.edge_index, pos_data.edge_attr, pos_data.smiles, pos_data.batch, False)
		positives.append(pos_z)
	  
	negative_sampler = BatchSampler(torch.flatten(negative_indices).tolist(), n, drop_last = False)
	negative_dataloader = DataLoader(SIDER,batch_sampler = negative_sampler)
	for neg_data in negative_dataloader:
		neg_data.to(device)
		_, neg_z = model(neg_data.x.float(),neg_data.edge_index, neg_data.edge_attr, neg_data.smiles, neg_data.batch, False)
		negatives.append(neg_z)
	#print(f"anchors1:{anchors}")
	  
	#negatives = [model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True) for data in dataset[negative_indices]]
	_,anchors_z = model( data.x.float(),data.edge_index, data.edge_attr, data.smiles, data.batch, False)
	return infoNCE(anchors_z, positives, negatives, device )



def train(model, data_loader, target_col, unsupervised_weight, device, optimizer, use_SSL=True,  **kwargs):
	model.train()
	u_loss_lst = []
	s_loss_lst = []
	t_loss_lst = []
	for i,  data in enumerate(data_loader):

		# ===replace column y
		data.y = data.y[:,target_col]
		data.to(device)
	
		#print(f"smi:{data.smiles}\nx:\n{data.x}\n edge_index:\n{data.edge_index}\n edge_attr:{data.edge_attr}")
		# === filter out data that does not have y label (i.e. y label is Nan)
		#data_lst = data.to_data_list()#~torch.isnan(data.y)]
		#data_lst = list(compress(data_lst, list(~torch.isnan(data.y).cpu().numpy())))


		#if(len(data_lst)!=0): # === if there is labeled data in this batch, then get the supervised loss, otherwise the loss is 0

		#	data = Batch.from_data_list(data_lst).to(device)
		#	out, z  = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True)# use our own x and edge_attr instead of data.x and data.edge_attr
		#	out = out.view(len(data.y))

		#	loss = BCELoss()(out, data.y)
		#else:
		#	loss = torch.tensor(0)

		out, z  = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True)# use our own x and edge_attr instead of data.x and data.edge_attr
		out = out.view(len(data.y))
		loss = BCELoss_no_NaN(out, data.y)
		
		# === unsupervised learning
		if (loss.isnan()):
			pass
			#print("loss is nan")
			#print(f"out:\n{out}, data.y[:,target_col]:\n{data.y[:,target_col]} ")
		else:
			if use_SSL == True:
				#unsupervised_loss = get_unsupervised_loss(method = 'pi_model', model = model, data = data, target_col = target_col) 
				unsupervised_loss = get_infoNCE(data, model, 5, device)#get_unsupervised_loss(method = 'pi_model',  model = model, data = data, target_col = target_col, edge_dropout_rate =kwargs['edge_dropout_rate'], device = device) 
				total_loss = loss + unsupervised_weight * unsupervised_loss
				u_loss_lst.append(unsupervised_weight*unsupervised_loss.item())
				s_loss_lst.append(loss.item())
				t_loss_lst.append(total_loss.item())
				#print(f"u_loss:{unsupervised_weight* unsupervised_loss:8.4f} || s_loss:{loss:8.4f} || t_loss:{total_loss:8.4f} || -- ori_u_loss:{unsupervised_loss:8.4f}  unsupervised_weight:{unsupervised_weight}")
			else:
				total_loss = loss
				t_loss_lst.append(total_loss.item())
				#print(f"t_loss:{total_loss}")
		
			total_loss.backward()
			optimizer.step()
			optimizer.zero_grad()



	#print(f"---------------------------------------------------------------")
	if use_SSL == True:		
		u_loss = mean(u_loss_lst)
		s_loss = mean(s_loss_lst)
		t_loss = mean(t_loss_lst)
		#print(f"u_loss:{u_loss:8.4f} || s_loss:{s_loss:8.4f} || t_loss:{t_loss:8.4f}")
	else:		
		t_loss = mean(t_loss_lst)
		#print(f"t_loss:{t_loss:8.4f}")

	return t_loss
