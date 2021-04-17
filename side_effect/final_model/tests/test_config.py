from config_parser import set_config_file, get_config

set_config_file('config.cfg')
file1 = get_config('cfg','lst')
#file1 = get_config('file','file1')
#file2 = get_config('file','file2')
print(type(file1))






