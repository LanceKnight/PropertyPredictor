import sys
import datetime

def tee_print(output_fstring, output_file, new_file):
	'''
	output_fstring: needs to be fstring
	output_file: name of the output_file
	new_file: boolean, whether to start a new file
	'''
	print(output_fstring)
	if(new_file == True):
		with open(output_file,'w') as f:
			print({datetime.datetime.now()}, file=f)
			print(output_fstring, file = f)
	else:
		with open(output_file, 'a') as f:
			print(output_fstring, file = f)
		
	
