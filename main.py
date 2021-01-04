import torch
import random
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


## parameters
#num_subsample=10000 # num of subsample
#byproduct_cutoff=10 # if the smiles string length is less than this number, it will be classified as byproduct; otherwise as main products
#data_type = torch.float
#training_perc = 0.8 # percentage of training data in the dataset
#val_perc = 0.1 # percentage of validation data in the dataset
#test_perc = 1-training_perc - val_perc # percentage of testing data in the dataset

# mol.py
from torch_geometric.utils import convert
import rdkit.Chem as Chem

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

def smiles2A(smiles, idxfunc=lambda x:x.GetIdx()):
	 '''convert smiles to adjacency matrix'''

	

def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):	
	mol = Chem.MolFromSmiles(r)
	if not mol:
		print(f"Error: cannot convert SMILES to mol. SMIILES:{r}")

	num_atoms = mol.GetNumAtoms()
	x = np.zeros((num_atoms, atom_fdim))

	# get x
	for atom in mol.GetAtoms():
		atom_feature = get_atom_features(atom)
		idx = idxfunc(atom)
		if idx >= num_atoms:
			print(f"Error: atom idx is larger than the number of atoms. SMILES:{r}, atom:{atom.GetSymbol()}")
		x[idx] = atom_feature

	# get edge_index
   A = Chem.rdmolops.GetAdjacencyMatrix(mol) # get adjacency matrix
	if A == None:
		print(f"Error: cannot get adjacency matrix of SMILES. SMILES:{r}")
	sA = sparse.csr_matrix(A) # convert A from numpy.matrix to scipy sparse matrix
	edge_index, edge_weight = convert.from_scipy_sparse_matrix(sA) # convert from scipy sparse matrix to edge_index

	return x, edge_index, edge_attr
	


# main.py

raw_data = "./data/USPTO/trim_train.txt.proc"#"./data/USPTO/train.txt.proc"
processed_data = "./data/USPTO/processed_training_data.pt"

batch_size = 20

def read_data(path):	
	rex_list, edit_list = [],[]
	with open(path, 'r') as f:
		for line in f:
			r,e = line.strip("\r\n ").split() # get reactants and edits from each line in the input file
			
						

			A = smiles2graph(r)

			if(A!=None):
				sA = sparse.csr_matrix(A)# convert A from numpy.matrix to scipy sparse matrix
				edge_index, edge_weight=convert.from_scipy_sparse_matrix(sA) # convert from scipy sparse matrix to edge_index
			else:
				print(f"Error:smiles2A is None, SMILES:{r}")
			data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = y)
	
	return 
	

def count(s):
    c = 0
    for i in range(len(s)):
        if s[i] == ':':
            c += 1
    return c

# load data
bucket_size = [10,20,30,40,50,60,80,100,120,150]
buckets = [[] for i in range(len(bucket_size))]


for i in range(len(buckets)):
	random.shuffle(buckets[i])

head = [0]*len(buckets)
print(f"head:{head}")
#print(f"buckets:{buckets}")
avil_buckets = [i for i in range(len(buckets)) if len(buckets[i]) > 0]
print(f"avil_buckets:{avil_buckets}")

src_batch, edit_batch=[],[]
bid = random.choice(avil_buckets)
print(f"bid:{bid}")
bucket= buckets[bid]
it=head[bid]
print(f"it:{it}")
data_len=len(bucket)
print(f"data_len:{data_len}")
for i in range(batch_size):
	react = bucket[it][0].split('>')[0]
	#print(f"react:{react},bucket[it]:{bucket[it]}")
	src_batch.append(react)
	edits=bucket[it][1]
	edit_batch.append(edits)
	print(f"before it:{it}")
	it=(it+1) % data_len
	print(f"after it:{it}")
head[bid]=it




class RexDatabase(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(InMemoryDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_path[0])

	@property
	def raw_dir(self):
		return [raw_data]

	@property
	def processeed_dir(self):
		return [processed_data]

	
	def process(self):


   	A = SMILES2Adjacency(product_smile) # get adjacency matrix of the product, in numpy.matrix	

		torch.save((self.data, self.slices),self.process_path[0])
