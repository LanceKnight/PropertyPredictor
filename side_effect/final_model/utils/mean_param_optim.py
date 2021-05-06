'''
put gather_param_optim in one file
'''

import sys
import os
import glob
import pandas as pd
import numpy as np

path = sys.argv[1]

folder_name = os.path.basename(os.path.normpath(path))

df =None

train_auc_array=[]
val_auc_array=[]
test_auc_array=[]

for i, param_file in enumerate(sorted(glob.glob(path+'/param_optim*'))):	
	print(param_file)
	param_df = pd.read_csv(param_file)
	if i ==0:
		df = param_df
	
	train_lst = param_df['train_auc'].tolist()
	val_lst = param_df['val_auc'].tolist()
	test_lst = param_df['test_auc'].tolist()
	train_auc_array.append(train_lst)
	val_auc_array.append(val_lst)
	test_auc_array.append(test_lst)

train_auc_array = np.array(train_auc_array)
val_auc_array = np.array(val_auc_array)
test_auc_array = np.array(test_auc_array)


train_auc = np.around(np.mean(train_auc_array, axis =0),4)
val_auc = np.around(np.mean(val_auc_array, axis =0),4)
test_auc = np.round(np.mean(test_auc_array, axis =0),4)

train_auc_sd = np.around(np.std(train_auc_array, axis=0),4)
val_auc_sd = np.around(np.std(val_auc_array, axis = 0),4)
test_auc_sd = np.around(np.std(test_auc_array, axis =0),4)


print(f"train_auc:\n{train_auc}\nval:{val_auc}\ntest:{test_auc}")
df['train_auc'] = train_auc
df['train_auc_sd'] = train_auc_sd

df['val_auc'] = val_auc
df['val_auc_sd'] = val_auc_sd

df['test_auc'] = test_auc
df['test_auc_sd'] = test_auc_sd

df.to_csv(f"param_optim/mean-{folder_name}.csv")
