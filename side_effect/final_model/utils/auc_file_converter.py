import sys
import pandas as pd
import glob
from statistics import mean
import os
'''
first argument is the folder for files
'''
path = sys.argv[1]

headers = ['a', 'b', 'c', 'd', 'e', 'mean']
def val_idx_generator():
	output = []
	last =6
	new = last 
	output.append(new)
	for i in range(35):
		if(i+1)%12==0:
			new = last +6
		else:
			if (i+1)%3 == 0:
				
				new = last +5
			else: 
				new = last +4
		last = new
		output.append(new)
	return output
		
val_idx_lst = [6]#val_idx_generator()#[3*(x+1) for x in range(1)]
index = ['train','val','test']*len(val_idx_lst)
df = pd.DataFrame(columns = headers, index=index)
col_value = []
new_headers = []
mean_lst = []
folder_name = os.path.basename(os.path.normpath(path))
#print(folder_name)

for i, auc_file in enumerate(sorted(glob.glob(path+'auc_file*'))):	
	print(auc_file)
	auc_df = pd.read_csv(auc_file)
	for val_idx in val_idx_lst:
		col_value.append(auc_df['train'][val_idx])
		col_value.append(auc_df['val'][val_idx])
		col_value.append(auc_df['test'][val_idx])
		#print(col_value)
	df[headers[i]] = col_value
	col_value = []
	#print(col_value)
	new_headers.append(auc_file[-5])
#print(df)

for row in df.itertuples():
	row = list(row)[1:-1]
	row = list(map(float, row))
	#print(row)
	m = round(mean(row),4)
	mean_lst.append(m)

new_headers.append('mean')
df.columns = new_headers
df['mean'] = mean_lst
print(df)
df.to_csv('auc_file_output/converter_output_'+folder_name+'.csv')
