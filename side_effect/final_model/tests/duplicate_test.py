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

read_lst = []
    
    
for i, smi in tqdm(enumerate(SMILES)):
	if(i%100)==0:
		print(f"i:{i}")
	mol = Chem.MolFromSmiles(smi)
	if mol is None:
		#print(i, smi)
		mol_error_count+=1
	else:
		smi = Chem.MolToSmiles(mol)
		for j, old_smi in enumerate(read_lst):
			if check_dup(smi, old_smi):
				duplicate_count+=1
				print(f"i:{i}, smi:{smi}  j:{j}, old_smi:{old_smi}")
		read_lst.append(smi)
print(f"# of duplicates:{duplicate_count}")
