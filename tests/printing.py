import sys
import datetime

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
		
	
