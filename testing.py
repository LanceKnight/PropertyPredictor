from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from statistics import mean
from itertools import compress


from molecule_processing import batch2attributes
from training import BCELoss_no_NaN, get_unsupervised_loss


def roc_auc_score_one_class_compatible(y_true, y_predict):
	sc = 9999
	try:
		sc = roc_auc_score(y_true, y_predict)
	except:
		print(f"error in running roc_auc_score(y_true, y_predict)   y_true:\n{y_true}, y_predict:\n{y_predict} sc:{sc}")
	return sc 
def pr_auc(y_true, y_predict):
	precision, recall, _ = precision_recall_curve(y_true, y_predict)
	auc_score = auc(recall, precision)	
	return auc_score

def test(model, data_loader,  target_col, device):
	model.eval()

	auc_lst = []
	loss_lst = []
	for data in data_loader:
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)	

		out, z = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, is_supervised = True) # use our own x and edge_attr instead of data.x and data.edge_attr

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
		#loss = get_loss(out,y)
		#print(f"loss:{loss}")
		#loss_lst.append(loss)

		#print(f"data.y.shape:{y}   out.shape:{out})")
		if ((len(y)!=0)  and (len(set(y))!=1)):
			sc = roc_auc_score_one_class_compatible(y, out)
			#sc = pr_auc(y, out)
			auc_lst.append(sc)
	try:
		return mean(auc_lst)#, mean(loss_lst)
	except:
		print(f"auc_lst:{auc_lst}")




def get_loss(out, y):
	u_loss_lst = []
	s_loss_lst = []
	t_loss_lst = []
	return  BCELoss_no_NaN(out, y)

	for i,  data in enumerate(data_loader):

		# ===replace column y
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		data.x = x
		data.edge_attr = edge_attr
		print(f"x:{x}, edge_attr:{edge_attr}")
		data.to(device)
	
		# === filter out data that does not have y label (i.e. y label is Nan)
#		data_lst = data.to_data_list()#~torch.isnan(data.y)]
#		data_lst = list(compress(data_lst, list(~torch.isnan(data.y).cpu().numpy())))


	#	if(len(data_lst)!=0): # === if there is labeled data in this batch, then get the supervised loss, otherwise the loss is 0

	#		data = Batch.from_data_list(data_lst).to(device)
	#		out, z  = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True)# use our own x and edge_attr instead of data.x and data.edge_attr
	#		out = out.view(len(data.y))

	#		loss = BCELoss()(out, data.y)
	#	else:
	#		loss = torch.tensor(0)

		#out, z  = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, True)# use our own x and edge_attr instead of data.x and data.edge_attr
		#out = out.view(len(data.y))
		loss = BCELoss_no_NaN(out, y)
		
		# === unsupervised learning
		if (loss.isnan()):
			pass
			#print("loss is nan")
			#print(f"out:\n{out}, data.y[:,target_col]:\n{data.y[:,target_col]} ")
		else:
			if use_SSL == True:
				#unsupervised_loss = get_unsupervised_loss(method = 'pi_model', model = model, data = data, target_col = target_col) 
				unsupervised_loss = get_unsupervised_loss(method = 'pi_model',  model = model, data = data, target_col = target_col, edge_dropout_rate =kwargs['edge_dropout_rate'], device = device) 
				total_loss = loss + unsupervised_weight * unsupervised_loss
				u_loss_lst.append(unsupervised_weight*unsupervised_loss.item())
				s_loss_lst.append(loss.item())
				t_loss_lst.append(total_loss.item())
				#print(f"u_loss:{unsupervised_weight* unsupervised_loss:8.4f} || s_loss:{loss:8.4f} || t_loss:{total_loss:8.4f} || -- ori_u_loss:{unsupervised_loss:8.4f}  unsupervised_weight:{unsupervised_weight}")
			else:
				total_loss = loss
				t_loss_lst.append(total_loss.item())
				#print(f"t_loss:{total_loss}")
		
	if use_SSL == True:		
		u_loss = mean(u_loss_lst)
		s_loss = mean(s_loss_lst)
		t_loss = mean(t_loss_lst)
	else:		
		t_loss = mean(t_loss_lst)
	return t_loss
