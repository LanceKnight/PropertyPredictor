import sys
import datetime
import pandas as pd

new_file = True
output_file =[]
def set_output_file(file_name):
	global output_file
	output_file = file_name

def tee_print(output_fstring):
	'''
	output_fstring: needs to be fstring
	output_file: name of the output_file
	new_file: boolean, whether to start a new file
	'''
	global new_file
	global output_file
	if(output_file == []):
		raise Exception("output_file is not set")

	if(new_file == True):
		with open(output_file,'w') as f:
			print(f"{datetime.datetime.now()}", file=f)
			print(output_fstring, file = f)
			print(output_fstring)
		new_file = False
	else:
		with open(output_file, 'a') as f:
			print(output_fstring, file = f)
			print(output_fstring)
		
def print_val_test_auc(train_auc, val_auc, test_auc, final_auc_file):	
	df = pd.DataFrame()
	#print(f"val_auc:{val_auc}, len:{len(val_auc)}")
	#print(f"test_auc:{test_auc}, len:{len(test_auc)}")
	df = df.append(train_auc)
	df = df.append(val_auc)
	df = df.append(test_auc)
	df.to_csv(final_auc_file)	
