import configparser


config = configparser.SafeConfigParser()
config_file = []

def Set_Config_File(file_name):
	global config_file 
	config_file= file_name

def Get_Config(section, option):
	with open(config_file, 'r') as f:
		config.readfp(f)
	value = config._sections[section][option]
	if (value[0] == "[") and (value[-1] == "]"):
		return eval(value)
	elif ('#' in value):
		value = value[:value.index('#')]
		return value
	else:
		return value		
