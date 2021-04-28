
import pandas as pd


param_set = None

def generate_param_set(params):
	global param_set

	df = pd.DataFrame()
	df['_tmpkey'] = [1]
	print(type(df))
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
	return df

def get_param_set(id):
	global param_set
	return param_set.loc[id]
	

