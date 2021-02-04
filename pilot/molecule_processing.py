import numpy as np
import torch
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges

def batch2attributes(smiles_batch, molecular_attributes=False):
	x = []
	edge_attr =[]
	for smi in smiles_batch:
		#print(f"smi:{smi}")
		smi_x, smi_edge_attr = smiles2attributes(smi, molecular_attributes= molecular_attributes)
		#print()
		#print(f"internal x:{smi_x}, edge_attr:{smi_edge_attr}")
		#print()
		x += smi_x
		edge_attr += smi_edge_attr
	return torch.tensor(x), torch.tensor(edge_attr)

def smiles2attributes(smiles, molecular_attributes=False):
	mol = MolFromSmiles(smiles)

	x = mol2x(mol, molecular_attributes)	
	edge_attr = mol2edge_attr(mol)
	return x, edge_attr

def mol2x(rdmol, molecular_attributes):

	attributes = [[] for i in rdmol.GetAtoms()]
	if molecular_attributes:
		labels = []
		[attributes[i].append(x[0]) \
			for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
		labels.append('Crippen contribution to logp')

		[attributes[i].append(x[1]) \
			for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
		labels.append('Crippen contribution to mr')

		[attributes[i].append(x) \
			for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(rdmol))]
		labels.append('TPSA contribution')

		[attributes[i].append(x) \
			for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(rdmol)[0])]
		labels.append('Labute ASA contribution')

		[attributes[i].append(x) \
			for (i, x) in enumerate(EState.EStateIndices(rdmol))]
		labels.append('EState Index')

		rdPartialCharges.ComputeGasteigerCharges(rdmol)
		[attributes[i].append(float(a.GetProp('_GasteigerCharge'))) \
			for (i, a) in enumerate(rdmol.GetAtoms())]
		labels.append('Gasteiger partial charge')

		# Gasteiger partial charges sometimes gives NaN
		for i in range(len(attributes)):
			if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
				attributes[i][-1] = 0.0

		[attributes[i].append(float(a.GetProp('_GasteigerHCharge'))) \
			for (i, a) in enumerate(rdmol.GetAtoms())]
		labels.append('Gasteiger hydrogen partial charge')

		# Gasteiger partial charges sometimes gives NaN
		for i in range(len(attributes)):
			if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
				attributes[i][-1] = 0.0

	x_list = []

	for i, atom in enumerate(rdmol.GetAtoms()):
		atom_attr = atom_attributes(atom, extra_attributes = attributes[i])
		x_list.append(atom_attr)
#	x_array = np.array(x_list)
#	x = torch.from_numpy(x_array)
	#return x
	return x_list

def mol2edge_attr(mol):
	attr_list = []
	for bond in mol.GetBonds():
		bond_attr = bond_attributes(bond)
		attr_list.append(bond_attr)
#	attr_array = np.array(attr_list)
#	edge_attr = torch.from_numpy(attr_array)
#	return edge_attr
	return attr_list
	

def bond_attributes(bond):
	# Initialize
	attributes = []
	# Add bond type
	attributes += one_hot_embedding(
		bond.GetBondTypeAsDouble(),
		[1.0, 1.5, 2.0, 3.0]
	)
	# Add if is aromatic
	attributes.append(bond.GetIsAromatic())
	# Add if bond is conjugated
	attributes.append(bond.GetIsConjugated())
	# Add if bond is part of ring
	attributes.append(bond.IsInRing())

	# NEED THIS FOR TENSOR REPRESENTATION - 1 IF THERE IS A BOND
	#attributes.append(1)

	#return np.array(attributes, dtype = np.single)
	return attributes

def atom_attributes(atom, extra_attributes=[]):
	attributes = []
	attributes += one_hot_embedding(
		atom.GetAtomicNum(), 
		[5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
	)
	
	# Add heavy neighbor count
	attributes += one_hot_embedding(
		len(atom.GetNeighbors()),
		[0, 1, 2, 3, 4, 5]
	)
	# Add hydrogen count
	attributes += one_hot_embedding(
		atom.GetTotalNumHs(),
		[0, 1, 2, 3, 4]
	)
	# Add formal charge
	attributes.append(atom.GetFormalCharge())
	# Add boolean if in ring
	attributes.append(atom.IsInRing())
	# Add boolean if aromatic atom
	attributes.append(atom.GetIsAromatic())


	attributes += extra_attributes

	#return np.array(attributes, dtype = np.single)
	return attributes	

def one_hot_embedding(val, lst):
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)


if __name__ == "__main__":	
	smi = 'CO'
	x, edge_attr = smiles2attributes(smi, molecular_attributes= True)
	#print(f'x:{x}, edge_attr:{edge_attr}' )
	smi_batch = ['CO', 'CN']
	x , edge_attr = batch2attributes(smi_batch, molecular_attributes = True)
	
	print(f'x:{x}, edge_attr:{edge_attr}' )
