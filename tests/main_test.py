from network_test import net
import torch
from torch.nn import MSELoss
import torch.optim as optim

model = net()

x = [[x] for x in range(100)]
x = torch.tensor(x, dtype = torch.float)
y = [[x**2] for x in range(100)]
y = torch.tensor(y, dtype = torch.float)
optimizer = optim.Adam()
def train():
	loss_lst=[]
	for i in range(len(x)):
		pred = model(x[i])
		#print(pred.shape)
		#print(y[i])

		loss_func = MSELoss()
		loss = loss_func(pred, y[i])
		loss_lst.append()
		
		loss.backward()

def main(num):
	for n in range(num):
		train()

main(50)
