import param_grid_search as ps

params = {'k1':[1,2,3], 'k2':[4,5], 'k3':9, 'k4':'this is a string'}


df = ps.generate_param_set(params)
print(df)

for i in range(len(df.index)):
	a = ps.get_param_set(i)
	#print(a)
