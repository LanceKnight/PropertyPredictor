from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from statistics import mean
from itertools import compress
from termcolor import cprint

from molecule_processing import batch2attributes
from training import BCELoss_no_NaN, get_loss


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

def test(model=None, data_loader=None,  target_col=None, device=None, method = None, unsupervised_weight = None, use_SSL = None, **kwargs):
	model.eval()

	auc_lst = []
	loss_lst = []
	supervised_loss_lst = []
	unsupervised_loss_lst = []
	for data in data_loader:
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)	

		out, z = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, is_supervised = True) # use our own x and edge_attr instead of data.x and data.edge_attr
		y = data.y[:,target_col]


		#print(f"y:{y}, id:{data.id}, auc_lst:{auc_lst}")
		if use_SSL:
			loss, supervised_loss, unsupervised_loss  = map(float,get_loss(method= method, data = data, model= model, predicted = (out,z),y = y, device = device, use_SSL=use_SSL, unsupervised_weight = unsupervised_weight, **kwargs))
			supervised_loss_lst.append(supervised_loss)
			unsupervised_loss_lst.append(unsupervised_loss)
		else:
			loss = float(get_loss(method= method, data = data, model= model, predicted = (out,z),y = y, device = device, use_SSL=use_SSL, unsupervised_weight = unsupervised_weight, **kwargs))

		loss_lst.append(loss.item())
		#==========convert to numpy array
		out = out.view(len(out))	
		out = out.cpu().detach().numpy()
		#print(f"out:{out}")
		y = y.view(len(y)).cpu().detach().numpy()
		#print(f"y:{y}")
		#==========remove NaN
		out = out[~np.isnan(y)]
		y = y[~np.isnan(y)]
		#print(f"loss:{loss}")
		#print(f"processed_y:{y}, out:{out}")
		#print(f"data.y.shape:{y}   out.shape:{out})")
		if ((len(y)!=0)  and (len(set(y))!=1)):
			sc = roc_auc_score_one_class_compatible(y, out)
			#sc = pr_auc(y, out)
			auc_lst.append(sc)
			#cprint(f"-----sc:{sc}, auc_lst:{auc_lst}", 'red')

	#print(f"auc_lst:{auc_lst}")
	if use_SSL:
		try:
			return mean(auc_lst), mean(loss_lst), mean(supervised_loss_lst), mean(unsupervised_loss_lst)
		except Exception as e:
			print(f"error msg:{e}\n auc_lst:\n {auc_lst}\n loss_lst: {loss_lst}\n supervised_loss_lst:\n{supervised_loss_lst}\n unsupervised_loss_lst:{unsupervised_loss_lst}")
	else:
		try:
			return mean(auc_lst), mean(loss_lst)
		except Exception as e:
			print(f"error msg:{e}\n auc_lst:\n {auc_lst}\n loss_lst: {loss_lst}")




