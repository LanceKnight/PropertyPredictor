from config_parser import set_config_file, get_config

set_config_file('config.cfg')
#set_config_file('../inputs/comp/base-ssl/base-ssl-a.cfg')
file1 = get_config('file','output_file')
#file1 = get_config('file','file1')
#file2 = get_config('file','file2')
print(file1)






