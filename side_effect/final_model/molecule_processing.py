import numpy as np
import torch
from rdkit.Chem import MolFromSmiles
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges

num_node_features = 32
num_edge_features = 7

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
		attr_list.append(bond_attr) # add twice to account for the bidirectional edge nature of chemical bonds
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
	smi_batch = ['OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', 'Cc1occc1C(=O)Nc2ccccc2']
	#smi_batch = ['OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', 'Cc1occc1C(=O)Nc2ccccc2', 'CC(C)=CCCC(C)=CC(=O)', 'c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43', 'c1ccsc1', 'c2ccc1scnc1c2 ', 'Clc1cc(Cl)c(c(Cl)c1)c2c(Cl)cccc2Cl', 'CC12CCC3C(CCc4cc(O)ccc34)C2CCC1O', 'ClC4=C(Cl)C5(Cl)C3C1CC(C2OC12)C3C4(Cl)C5(Cl)Cl', 'COc5cc4OCC3Oc2c1CC(Oc1ccc2C(=O)C3c4cc5OC)C(C)=C ', 'O=C1CCCN1', 'Clc1ccc2ccccc2c1', 'CCCC=C', 'CCC1(C(=O)NCNC1=O)c2ccccc2', 'CCCCCCCCCCCCCC', 'CC(C)Cl', 'CCC(C)CO', 'N#Cc1ccccc1', 'CCOP(=S)(OCC)Oc1cc(C)nc(n1)C(C)C', 'CCCCCCCCCC(C)O', 'Clc1ccc(c(Cl)c1)c2c(Cl)ccc(Cl)c2Cl ', 'O=c2[nH]c1CCCc1c(=O)n2C3CCCCC3', 'CCOP(=S)(OCC)SCSCC', 'CCOc1ccc(NC(=O)C)cc1', 'CCN(CC)c1c(cc(c(N)c1N(=O)=O)C(F)(F)F)N(=O)=O', 'CCCCCCCO', 'Cn1c(=O)n(C)c2nc[nH]c2c1=O', 'CCCCC1(CC)C(=O)NC(=O)NC1=O', 'ClC(Cl)=C(c1ccc(Cl)cc1)c2ccc(Cl)cc2', 'CCCCCCCC(=O)OC', 'CCc1ccc(CC)cc1', 'CCOP(=S)(OCC)SCSC(C)(C)C', 'COC(=O)Nc1cccc(OC(=O)Nc2cccc(C)c2)c1', 'ClC(=C)Cl', 'Cc1cccc2c1Cc3ccccc32', 'CCCCC=O', 'N(c1ccccc1)c2ccccc2', 'CN(C)C(=O)SCCCCOc1ccccc1', 'CCCOP(=S)(OCCC)SCC(=O)N1CCCCC1C', 'CCCCCCCI', 'c1c(Cl)cccc1c2ccccc2', 'OCCCC=C', 'O=C2NC(=O)C1(CCC1)C(=O)N2', 'CC(C)C1CCC(C)CC1O ', 'CC(C)OC=O', 'CCCCCC(C)O', 'CC(=O)Nc1ccc(Br)cc1', 'c1ccccc1n2ncc(N)c(Br)c2(=O)', 'COC(=O)C1=C(C)NC(=C(C1c2ccccc2N(=O)=O)C(=O)OC)C ', 'c2c(C)cc1nc(C)ccc1c2 ', 'CCCCCCC#C', 'CCC1(C(=O)NC(=O)NC1=O)C2=CCCCC2 ', 'c1ccc2c(c1)ccc3c4ccccc4ccc23', 'CCC(C)n1c(=O)[nH]c(C)c(Br)c1=O ', 'Clc1cccc(c1Cl)c2c(Cl)c(Cl)cc(Cl)c2Cl ', 'Cc1ccccc1O', 'CC(C)CCC(C)(C)C', 'Cc1ccc(C)c2ccccc12', 'Cc1cc2c3ccccc3ccc2c4ccccc14', 'CCCC(=O)C', 'Clc1cc(Cl)c(Cl)c(c1Cl)c2c(Cl)c(Cl)cc(Cl)c2Cl ', 'CCCOC(=O)CC', 'CC34CC(O)C1(F)C(CCC2=CC(=O)C=CC12C)C3CC(O)C4(O)C(=O)CO', 'Nc1ccc(O)cc1']
	x , edge_attr = batch2attributes(smi_batch, molecular_attributes = True)
	
	print(f'x:{x}, edge_attr:{edge_attr.shape}' )
