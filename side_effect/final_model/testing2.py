from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from statistics import mean

from molecule_processing import batch2attributes


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

def test(model, data_loader, target_col, device):
	model.eval()

	auc_lst = []
	for data in data_loader:
		x, edge_attr = batch2attributes(data.smiles, molecular_attributes= True)
		data.x = x
		data.edge_attr = edge_attr
		data.to(device)	

		out = model(data.x.float(), data.edge_index, data.edge_attr, data.smiles, data.batch, is_supervised = True) # use our own x and edge_attr instead of data.x and data.edge_attr

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

		#logger.report_matrix("Confusion Matrix", "value", iteration = )


		#print(f"data.y.shape:{y}   out.shape:{out})")
		if ((len(y)!=0)  and (len(set(y))!=1)):
			sc = roc_auc_score_one_class_compatible(y, out)
			#sc = pr_auc(y, out)
			auc_lst.append(sc)
	try:
		return mean(auc_lst)
	except:
		print(f"auc_lst:{auc_lst}")
