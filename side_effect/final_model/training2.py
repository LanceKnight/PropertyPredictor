import torch
from torch_geometric.utils import dropout_adj

from statistics import mean



from molecule_processing import batch2attributes

def BCELoss_no_NaN(out, target):
	#print(f"out.shape:{out.shape}             target.shape:{target.shape}")
	#target_no_NaN = torch.where(torch.isnan(target), out, target)
	target_no_NaN = target[~torch.isnan(target)]
	out = out[~torch.isnan(target)]
	target_no_NaN = target_no_NaN.detach() 
	#print(f"target_no_NaN:{target_no_NaN}")
	return torch.nn.BCELoss()(out, target_no_NaN)

def get_unsupervised_loss(method=None, **kwargs):
	loss = torch.tensor([0])
	if method == 'pi_model':
		model = kwargs['model']
		data = kwargs['data']
		edge_dropout_rate = kwargs['edge_dropout_rate']
		target_col = kwargs['target_col']
		#print(f"edge_index:{data.edge_index.shape} edge_attr:{data.edge_attr.shape}")
		edge_index2, edge_attr2 =  dropout_adj(data.edge_index, data.edge_attr, p = edge_dropout_rate)
		#edge_index3, edge_attr3 =  dropout_adj(data.edge_index, data.edge_attr, p = edge_dropout_rate)
		#print(f"edge_index:{edge_index2.shape} edge_attr:{edge_attr2.shape}")
		#print(f"edge_index:{edge_index3.shape} edge_attr:{edge_attr3.shape}")
		out2 = model(data.x.float(), edge_index2, edge_attr2, data.smiles, data.batch, False)# use our own x and edge_attr instead of data.x and data.edge_attr
		out2 = out2.view(len(data.y[:,target_col]))

		out3 = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, False)# use our own x and edge_attr instead of data.x and data.edge_attr
		out3 = out3.view(len(data.y[:,target_col]))
		loss = torch.nn.MSELoss()(out3, out2)
	elif method == 'edge_dropout':
		pass	
	else:
		print('cannot get unsupervised loss!')

	return loss

def train(model, data_loader, target_col, unsupervised_weight, device, optimizer, use_SSL=True, debug_mode=False, **kwargs):
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
		if (loss.isnan()):
			pass
			#print("loss is nan")
			#print(f"out:\n{out}, data.y[:,target_col]:\n{data.y[:,target_col]} ")
		else:
			if use_SSL == True:
				#unsupervised_loss = get_unsupervised_loss(method = 'pi_model', model = model, data = data, target_col = target_col) 
				unsupervised_loss = get_unsupervised_loss(method = 'pi_model', model = model, data = data, target_col = target_col, edge_dropout_rate =kwargs['edge_dropout_rate']) 
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


		if(debug_mode):
			out_list = out.cpu().detach().numpy()
			y_list = data.y.cpu().detach().numpy()
			#print(f"{len(out_list)}, {len(y_list)}")
			for i in range(len(out_list)): 
				print(f"{out_list[i][0]}, {y_list[i][0]}") # for making correlation plot

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
