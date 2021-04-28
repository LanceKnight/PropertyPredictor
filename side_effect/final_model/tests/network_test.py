import torch.nn
from torch.nn import Linear

class net(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.lin1 = Linear(1,1)
	def forward(self,x):
		return self.lin1(x)

