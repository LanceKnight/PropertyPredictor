import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader

epoches = 100

ESOL_dataset = MoleculeNet(root = "../data/raw/ESOL", name = "ESOL")
print("data info:")
print("============")
print(f"num of data:{len(ESOL_dataset)}")

num_data = len(ESOL_dataset)
train_num = int(num_data * 0.64)
val_num = int(num_data * 0.16)
test_num = num_data - train_num - val_num
print(f"train_num = {train_num}, val_num = {val_num}, test_num = {test_num}")

train_loader = DataLoader(ESOL_dataset[:train_num], batch_size = 64, shuffle = False)
validate_loader = DataLoader(ESOL_dataset[train_num:train_num+val_num], batch_size = 64, shuffle = False)
test_loader = DataLoader(ESOL_dataset[-test_num:], batch_size = 64, shuffle = False)

class MyNet(torch.nn.Module):
	def __init__(self, num_features):
		super(MyNet, self).__init__()
		self.conv1 = GCNConv(num_features, 16)
		self.conv2 = GCNConv(16, 1)
	def forward(self, x, edge_index, batch):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)
		x = global_add_pool(x, batch)
		return x
		
is_cuda = torch.cuda.is_available()
print(f"is_cuda:{is_cuda}")

device = torch.device('cuda' if is_cuda else 'cpu')
model = MyNet(ESOL_dataset.num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
criterion = torch.nn.MSELoss()

#example
data = ESOL_dataset[0].to(device)
print(f"data[0]:{data}")

def train(data_loader):
	model.train()
	for data in data_loader:
		data.to(device)
		out = model(data.x.float(), data.edge_index, data.batch)
		#print(f"data:{data}")
		#print(f"out:{len(out)},y:{len(data.y)}")
		loss = criterion(out, data.y)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()


def test(data_loader):
	model.eval()

	squared_error_sum = 0 
	for data in data_loader:
		data.to(device)
		out = model(data.x.float(), data.edge_index, data.batch)
		pred = out
		#print(f"pred:{len(pred)},data.y:{len(data.y)}")
		t = sum(pow((pred - data.y),2)).cpu().detach().numpy()
		squared_error_sum +=t
	
	MSE = squared_error_sum / len(data_loader)
	return MSE

for epoch in range(epoches):
	train(train_loader)
	train_MSE = test(train_loader)
	test_MSE = test(test_loader)
	print(f"Epoch:{epoch:03d}, Train MSE:{train_MSE}, Test MSE:{test_MSE}")
