import pandas as pd

headers = ['','h1','h2']
#headers = ['','', 'w=0.01','target0','target1','target2','target3','target4','target5','target6','','', 'w=0.1','target0','target1','target2','target3','target4','target5','target6','','', 'w=1','target0','target1','target2','target3','target4','target5','target6','','', 'w=10','target0','target1','target2','target3','target4','target5','target6','','', 'w=100','target0','target1','target2','target3','target4','target5','target6','','', 'w=1000','target0','target1','target2','target3','target4','target5','target6',]



df = pd.DataFrame(columns=headers, index=None)
df.loc[len(df.index)] = [1,2,3]
print(df)
df.to_csv('output.csv')
