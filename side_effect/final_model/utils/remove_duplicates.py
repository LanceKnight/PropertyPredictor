import pandas as pd
import rdkit.Chem as Chem
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

df = pd.read_csv('../../../data/raw/SIDER/sider/raw/syn_SIDER_toxcast_tox21.csv')
SMILES = df['smiles']
#print(f"num of data:{SMILES}")

mol_error_count = 0
duplicate_count =0
def check_dup(smi_a, smi_b):
    if(smi_a==smi_b):
        return True
    else:
        return False

non_duplicate_lst = []
non_duplicate_idx = []    
SMILES = SMILES
 
for i, smi in tqdm(enumerate(SMILES)):
	found_duplicate = False
	if(i%100)==0:
		pass
		#print(f"i:{i}")
	mol = Chem.MolFromSmiles(smi)
	if mol is None:
		#print(i, smi)
		mol_error_count+=1
	else:
		smi = Chem.MolToSmiles(mol)
		for j, old_smi in enumerate(non_duplicate_lst):
			if check_dup(smi, old_smi):
				duplicate_count+=1
				#print(f"i:{i}, smi:{smi}  j:{j}, old_smi:{old_smi}")
				found_duplicate = True
				break	
		if found_duplicate:
			pass
		else:
			#print(f"{i} added")
			non_duplicate_lst.append(smi)
			non_duplicate_idx.append(i)
print(f"# of duplicates:{duplicate_count}, # of mol err: {mol_error_count}")



#print(f"non_duplicate_idx:{non_duplicate_idx}")
new_df = df.loc[non_duplicate_idx]
new_df.to_csv('non_duplicate.csv', index=False)

