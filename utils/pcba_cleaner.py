import pandas as pd

df = pd.read_csv('../../../data/raw/PCBA/pcba.csv')
df.head()
#df.dropna(subset='smiles', inplace=True)
print(len(df.index))


df = df['smiles']
df.to_csv('pcba.csv')

