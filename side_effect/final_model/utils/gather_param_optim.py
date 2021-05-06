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
	df = pd.concat([df, param_df])
	
df.to_csv(f"param_optim/gather-{folder_name}.csv")
