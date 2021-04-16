import sys
from printing import tee_print, set_output_file
import torch
from torch.nn import Sigmoid, Linear, BCELoss
import datetime
from statistics import mean


x = [[0.1], [0.3], [0.1], [1.2], [0.1], [1.8], [0.1]]
y_true = [  [1.0],   [0],  [1],   [0],  [1],   [0],   [1]]
x = torch.tensor(x)
y_true = torch.tensor(y_true)

class MyNet(torch.nn.Module):
	def __init__(self):
		super(MyNet, self).__init__()
		self.lin1 = Linear(1, 50)
		self.lin2 = Linear(50, 1)
	def forward(self, x):
		hid = self.lin1(x)
		out = self.lin2(hid)
		return Sigmoid()(out)


epochs = 100

def train(model, optimizer):
	model.train()
	loss = 0
	for i, sample in enumerate(x):
		y_pred = model(sample)
		#print(f"y_pred: {y_pred}  y_true:{y_true[i]}")
		loss = BCELoss()(y_pred, y_true[i])	
		loss.backward()		
		optimizer.step()
		optimizer.zero_grad()
	return loss
def test():
	model.eval()
	sc_list = []
	for i, sample in enumerate(x):
		y_pred = model(sample)
		sc = BCELoss()(y_pred, y_true[i])
		sc = sc.detach().numpy()
		sc_list.append(sc)

	return mean(list(sc_list))

for w in range(3):
	 
	model = MyNet()

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.1 )
	for epoch in range(epochs):
		sc = train(model, optimizer)
		#sc = test()
		if epoch %10 == 0:
			print(f"epoch:{epoch}  sc:{sc}")	
