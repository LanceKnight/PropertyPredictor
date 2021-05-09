import configparser


config = configparser.SafeConfigParser(interpolation = configparser.ExtendedInterpolation())
config_file = []

def set_config_file(file_name):
	global config_file 
	config_file= file_name

def get_config(section, *args):
	if(len(args) >0):
		option = args[0]
		with open(config_file, 'r') as f:
			config.readfp(f)
		value = config[section][option]
		if is_list(value):
			return eval(value)
		elif ('#' in value):
			value = value[:value.index('#')]
			return value
		else:
			return value
	else:
		with open(config_file, 'r') as f:
			config.readfp(f)
		value = config[section]
		return value

def is_list(value):
	return (value[0] == "[") and (value[-1] == "]")

def get_config_dict():
	result = {}
	with open(config_file, 'r') as f:
		config.readfp(f)

	for section in config.sections():
		for key, value in config.items(section):
			if is_list(value):
				value = eval(value)
			try:
				value =int(value)
			except:

				try:
					value = float(value)
				except:
					pass

				
			result[key] = value
	return result
	
