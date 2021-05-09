import sys
import pandas as pd

lr = [0.07, 0.007, 0.0007]


input_file = sys.argv[1]
input_df = pd.read_csv(input_file, header = None,float_precision='round_trip' )
output_df = pd.DataFrame()
row = input_df.loc[0]
#print(input_df)

#for value in row:
#	print(value)
row = row.tolist()

output = []
series = []
for i, value in enumerate(row):
	if (value in lr) and (i !=0):
		#print(value)
		output.append(series)
		series = []
	series.append(value)
output.append(series)

output_df = pd.DataFrame.from_records(output).transpose()
print(output_df)
output_df.to_csv('lr_converter_output/lr_output.csv')
	



