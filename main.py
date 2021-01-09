import torch
import random
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
import numpy as np
from scipy import sparse

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_feature_dim = len(elem_list) + 6 + 6 + 6 + 1
bond_feature_dim = 6

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    return np.array(one_hot_encoding(atom.GetSymbol(), elem_list) 
            + one_hot_encoding(atom.GetDegree(), [0,1,2,3,4,5]) 
            + one_hot_encoding(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + one_hot_encoding(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def get_bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

	

def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):	
	mol = Chem.MolFromSmiles(smiles)
	if not mol:
		raise ValueError(f"smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):smiles cannot be converted to mol, smiles:{smiles}")

	num_atoms = mol.GetNumAtoms()
	num_bonds = mol.GetNumBonds()
	x = np.zeros((num_atoms, atom_feature_dim))
#	edge_index = np.zeros((2, num_bonds))
	edge_attr = np.zeros((2*num_bonds, bond_feature_dim)) # in PyG, one bond is treated as two edges for both directions.

	# get x
	for atom in mol.GetAtoms():
		atom_feature = get_atom_features(atom)
		idx = idxfunc(atom)
		#print(f"atom id:{idx}")
		if idx >= num_atoms:
			print(f"Error: atom idx is larger than the number of atoms. SMILES:\n {r}\n atom:\n{atom.GetSymbol()}")
		x[idx] = atom_feature

	# get edge_index
	A = Chem.rdmolops.GetAdjacencyMatrix(mol) # get adjacency matrix
	if A is  None:
		print(f"Error: cannot get adjacency matrix of SMILES. SMILES:{r}")
	else:
		sA = sparse.csr_matrix(A) # convert A from numpy.matrix to scipy sparse matrix
		edge_index, edge_weight = convert.from_scipy_sparse_matrix(sA) # convert from scipy sparse matrix to edge_index

	# get edge_attr
	for bond in mol.GetBonds():
		idx = bond.GetIdx()
		#print(f"bond id:{idx}")
		edge_attr[idx] = edge_attr[num_bonds+idx] = get_bond_features(bond)
	return torch.from_numpy(x).float(), edge_index, torch.from_numpy(edge_attr).float()


#output_processing.py

changed_bond_to_index={0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
#bond_change_set = [0, 1, 2, 3, 1.5]
changed_bond_dim = len(changed_bond_to_index)
INVALID_BOND = -1
def get_bond_label(reactant_smiles, edits):
	mol = Chem.MolFromSmiles(reactant_smiles)
	num_atoms = mol.GetNumAtoms()
	changed_bond_map = np.zeros((num_atoms, num_atoms, changed_bond_dim))

	#create a tensor that store (atom1, atom2, changed_bond_type)
	for s in edits.split(';'):
		a1,a2,bo = s.split('-')
		x = min(int(a1)-1,int(a2)-1)
		y = max(int(a1)-1, int(a2)-1)
		z = changed_bond_to_index[float(bo)]
		changed_bond_map[x,y,z] = changed_bond_map[y,x,z] = 1

	#flatten changed_bond_map to an array
	labels = []
	for i in range(num_atoms):
		for j in range(num_atoms):
			for k in range(len(changed_bond_to_index)):
				if i == j:
					labels.append(INVALID_BOND) # mask
				else:
					labels.append(changed_bond_map[i,j,k])
	return np.array(labels)


#def get_bond_label(edge_index, edits):
#	num_edits = len(edits.split(";"))
#
#	y_index = np.zeros((2, 2*num_edits)) # y_index stores graph edges, instead of chemical bonds. one chemical bond counts as two edges, both directions. 
#	y_attr = np.zeros((2*num_edits, bond_label_dim))
#
#	bond_start_atoms = []
#	bond_end_atoms = []
#
#	for i, edit in enumerate(edits.split(";")):
#		print(f"i:{i}, edit:{edit}")
#		a1, a2, bond = edit.split('-')
#		print(f"a1:{a1}, a2:{a2}, bond:{bond}")
#		x = min(int(a1)-1, int(a2)-1)
#		y = max(int(a1)-1, int(a2)-1)
#		z = changed_bond_to_index[float(bond)]
#		bond_start_atoms.append(x)
#		bond_start_atoms.append(y)
#		bond_end_atoms.append(y)
#		bond_end_atoms.append(x)
#		print(f"bond:{float(bond)}")
#		y_attr[i] = one_hot_encoding(z,bond_change_set)
#		y_attr[num_edits+i] = one_hot_encoding(z, bond_change_set)
#
#	y_index[0] = bond_start_atoms
#	y_index[1] = bond_end_atoms
#	return y_index, y_attr


#def get_bond_label(reactant_smiles, edits):
#	mol = Chem.MolFromSmiles(reactant_smiles)
#	if not mol:
#		raise ValueError(f"get_bond_label(reactants_smiles, edits): reactants_smiles cannot be converted to mol, reactant_smiles:{reactant_smiles}")
#	num_atoms = mol.GetNumAtoms()
#	num_bonds = mol.GetNumBonds()
#	y_index = np.zeros((2, pow(num_atoms, 2)))
#	y_attr = np.zeros((num_bonds, bond_label_dim))
#
#	bond_start_atoms = []
#	bond_end_atoms = []
#	for edit in edits.split(';'):
#		a1,a2,bond = edit.split('-')
#		x = min(int(a1)-1,int(a2)-1)
#		y = max(int(a1)-1, int(a2)-1)
#		z = changed_bond_to_index[float(bond)]
#		bond_start_atoms.append(x)
#		bond_start_atoms.append(y)
#		bond_end_atoms.append(y)
#		bond_end_atoms.append(x)
#		y_index[0] = bond_start_atoms
#		y_index[1] = bond_end_end
#
#
#	return y_index, y_attr

	
# test
#line = "[CH2:15]([CH:16]([CH3:17])[CH3:18])[Mg+:19].[CH2:20]1[O:21][CH2:22][CH2:23][CH2:24]1.[Cl-:14].[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])[N:8]([O:9][CH3:10])[CH3:11])[cH:12][cH:13]1>>[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])[CH2:15][CH:16]([CH3:17])[CH3:18])[cH:12][cH:13]1 6-8-0.0;15-6-1.0;15-19-0.0"
line = "[C:1][C:2]>>[C:1].[C:2] 1-2-0.0"
line = "[C:1].[C:2]>>[C:1]=[C:2] 1-2-2.0"
#line = "C 6"

reactions, edits = line.strip("\r\n ").split()
reactants = reactions.split('>')[0]
print(f"reactants:\n{reactants}")
x, edge_index, edge_attr  = smiles2graph(reactants)
y_index = get_bond_label(reactants, edits)
print(f"x:\n{x}\n\
		  edge_index:\n{edge_index}\n\
		  edge_attr:\n{edge_attr}\n\
		  y_index:\n{y_index}"
		 # y_attr:\n{y_attr}"\
		)

#nn.py
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

#class TestConv(MessagePassing):
#	def __init__(self, in_size, out_size):
#		super(TestConv, self).__init__(aggr='add')
#		self.lin = Linear(in_size, out_size)
#
#test_conv = TestConv(10, 100)
#print(f"test_conv:{test_conv}")

#lin = Linear(82, 15).float()
#print(f"lin(x):{lin(x).dtype}")

class WL_update_conv(MessagePassing):
	def __init__(self, num_features, num_edge_attr, hidden_size): #x_size may not be used
		super(WL_update_conv, self).__init__(aggr='add')
		#self.self_lin = Linear(hidden_size, hidden_size).float()
		#self.edge_lin = Linear(edge_attr_size, hidden_size).float()
		self.lin1 = Linear(hidden_size + num_edge_attr, hidden_size).float()
		self.lin2 = Linear(2*hidden_size, hidden_size).float()
			
	def forward(self, x, edge_index, edge_attr):
		print(f"WLconv-x.shape:{x.shape}")
		print(f"WLconv-edge_index.shape:{edge_index.shape}")
		print(f"WLconv-edge_index:edge_attr.shape:{edge_attr.shape}")
		neighbor_sum = self.propagate(edge_index, x=x, edge_attr = edge_attr)
		print(f"neighbor_sum.shape:{neighbor_sum.shape}")
		print(f"x.shape:{x.shape}")
		new_label= self.lin2( torch.cat((neighbor_sum, x), dim = 1) ).relu()
		
		return new_label
	#def message(self, x_j, edge_attr):
	#	print(f"WLconv-message-x_j.shape:{x_j.shape}")
	#	print(f"WLconv-message-edge_attr:{edge_attr.shape}")
	#	neighbor_atoms = self.self_lin(x_j)
	#	print(f"WLconv-message-neighbor_atoms.shape: {neighbor_atoms.shape}")
	#	neighbor_bonds = self.edge_lin(edge_attr)
	#	print(f"WLconv-message-neighbor_bonds.shape: {neighbor_bonds.shape}")
	#	neighbor = neighbor_atoms * neighbor_bonds
	#	print("here2")
	#	return neighbor
	def message(self, x_i, x_j, edge_attr):
		print(f"WLconv-message-x_j.shape:{x_j.shape}")
		print(f"WLconv-message-edge_attr:{edge_attr.shape}")
		neighbor_hidden= self.lin1(torch.cat((x_j, edge_attr), dim = 1)).relu()
		print(f"WLconv-message-neibor_hidden:{neighbor_hidden.shape}")
	#	neighbor_sum = torch.sum(neighbor_hidden, dim = 0)
	#	print(f"neighbor_sum.shape:{neighbor_sum.shape}")
	#	print(f"x_i.shape:{x_i.shape}")
	#	t = torch.cat((neighbor_sum, x_i))
	#	print(f"cat.shape:{t}")
	#	new_label = self.lin2(t).relu()
		return neighbor_hidden
		


#model.py
import torch
from torch.nn import Linear
import torch.nn.functional as F
class WLN(torch.nn.Module):
	def __init__(self, input_size, edge_attr_size, hidden_size, depth):
		super(WLN, self).__init__()       
		self.lin = Linear(input_size, hidden_size, bias = False).float()
		self.WLconv1 = WL_update_conv(input_size, edge_attr_size,  hidden_size).float()
	def forward(self, x, edge_index, edge_attr):
		print(f"x.dtype:{x.dtype}")
		for param in self.lin.parameters():
			print(f"self.lin:{param}")
		x = self.lin(x)
		x = x.relu()
		print(f"WLN-x.shape:{x.shape}")
		for i in range(depth):
			x = self.WLconv1(x, edge_index, edge_attr)

		
		return x



# main.py
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader

raw_data = "./data/raw/USPTO/trim_train.txt.proc"#"./data/USPTO/train.txt.proc"
processed_data = "data.pt"

batch_size = 1

class RexDataset(InMemoryDataset):
	def __init__(self, root, input_file, transform=None, pre_transform=None):
		self.input_file = input_file
		super(RexDataset, self).__init__(root, transform, pre_transform)		
		self.data, self.slices = torch.load(self.processed_paths[0])
		self.a = 1
	@property
	def raw_file_names(self):
		return ['train.txt.proc']

	@property
	def processed_file_names(self):
		return [processed_data]

	def download(self):
		pass
		
	def process(self):	
		data_list = []
		#print(f"a:{self.a}")
		with open(self.input_file, 'r') as f:
			for line in f:
				r,e = line.strip("\r\n ").split() # get reactants and edits from each line in the input file			
				react = r.split('>')[0]
				x, edge_index, edge_attr = smiles2graph(react, idxfunc = lambda x:x.GetIntProp('molAtomMapNumber')-1)	
				#print(f"process:x: {x.dtype}")
				y = get_bond_label(react, e)
				one_data_point = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = y)
				data_list.append(one_data_point)
		data, slices = self.collate(data_list)
		torch.save((data, slices),self.processed_paths[0])



train_data = RexDataset("./data/my_data", raw_data)
print(f"first data point:{train_data[0]}")
train_data = train_data[:100]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
hidden_channels = 100
depth = 1

print(f"num_features:{train_data.num_features}      num_edge_features:{train_data.num_edge_features}                hidden_channels:{hidden_channels}")
model = WLN(train_data.num_features, train_data.num_edge_features, hidden_channels, depth).float()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)


def train(data_loader):
	model.train()
	optimizer.zero_grad()  # Clear gradients.
	for data in data_loader:
		out = model(data.x, data.edge_index, data.edge_attr)  # Perform a single forward pass.
		loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
		loss.backward()  # Derive gradients.
		optimizer.step()  # Update parameters based on gradients.
	return loss

train(train_loader)


#def read_data(path):	
#	rex_list, edit_list = [],[]
#	with open(path, 'r') as f:
#		for line in f:
#			r,e = line.strip("\r\n ").split() # get reactants and edits from each line in the input file			
#			react = r.split('>')[0]
##			data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = y)
#	
#	return 
#	
#
#def count(s):
#    c = 0
#    for i in range(len(s)):
#        if s[i] == ':':
#            c += 1
#    return c





# load data
#bucket_size = [10,20,30,40,50,60,80,100,120,150]
#buckets = [[] for i in range(len(bucket_size))]
#
#
#for i in range(len(buckets)):
#	random.shuffle(buckets[i])
#
#head = [0]*len(buckets)
#print(f"head:{head}")
##print(f"buckets:{buckets}")
#avil_buckets = [i for i in range(len(buckets)) if len(buckets[i]) > 0]
#print(f"avil_buckets:{avil_buckets}")
#
#src_batch, edit_batch=[],[]
#bid = random.choice(avil_buckets)
#print(f"bid:{bid}")
#bucket= buckets[bid]
#it=head[bid]
#print(f"it:{it}")
#data_len=len(bucket)
#print(f"data_len:{data_len}")
#for i in range(batch_size):
#	react = bucket[it][0].split('>')[0]
#	#print(f"react:{react},bucket[it]:{bucket[it]}")
#	src_batch.append(react)
#	edits=bucket[it][1]
#	edit_batch.append(edits)
#	print(f"before it:{it}")
#	it=(it+1) % data_len
#	print(f"after it:{it}")
#head[bid]=it




