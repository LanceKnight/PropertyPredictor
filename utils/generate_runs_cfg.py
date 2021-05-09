import sys
import argparse
import string
import shutil
from configparser import SafeConfigParser
import os
'''
generate n .cfg files ending a, b, c, d, e from the sample.cfg file

1. first argument is the sample.cfg
2. use -n to set the number of runs. default is 5
3. use -o to set the output folder name. default is to use the sample name as the folder name
'''

arg_parser = argparse.ArgumentParser(description = 'generate 5 .cfg files with differnt fold ending 0, 1, 2, 3, 4 from the sample.cfg file')
arg_parser.add_argument('sample')
arg_parser.add_argument('-n', default = 5)
arg_parser.add_argument('-o', default = '../inputs/comp')
args = vars(arg_parser.parse_args())
#print(args)

sample_path = args['sample']
sample = os.path.basename(os.path.normpath(sample_path))
sample_name = sample[:-4]
runs = int(args['n'])
output_folder = args['o']+'/'+sample_name
result_folder = '../results/comp/'+ sample_name

try:
	os.mkdir(output_folder)
	os.mkdir(result_folder)
except Exception as e:
	print('folder exists')
suffix = [x+1 for x in range(10)]
config_parser = SafeConfigParser()
for i in range(runs):
	new_name = sample_name+'-'+str(suffix[i])
	new_file = output_folder + '/'+new_name + '.cfg'
	shutil.copyfile(sample_path, new_file)
	config_parser.read(new_file)
	config_parser.set('file', 'file_name', new_name)
	config_parser.set('cfg','fold', str(suffix[i])) 
	with open(new_file, 'w') as f:
		config_parser.write(f)
print("done!")

