
import pandas as pd


param_set = None
output_file = None

def generate_param_sets(params):
	global param_set
	global output_file

	df = pd.DataFrame()
	df['_tmpkey'] = [1]
	headers = []
	for param in params:
		headers.append(param)
		value = params[param]
		if isinstance(value, list):
			new_df = pd.DataFrame({param:value})
		else:
			new_df = pd.DataFrame({value})
		new_df['_tmpkey'] = 1
		df = pd.merge(df, new_df, on='_tmpkey')#.drop('_tmpkey', axis=1)
	df = df.drop('_tmpkey', axis=1)
	df.columns = headers
	param_set = df
	
	output_file = params['param_optim_file']

	return df

def get_param_set(id):
	global param_set
	return param_set.loc[id]
	
def get_param_sets_length():
	global param_set
	return len(param_set.index)

def record_result(id, header, value):
	global param_set
	param_set.loc[id, header]=value

def save_file():
	global param_set
	global output_file
	param_set.to_csv(output_file)
