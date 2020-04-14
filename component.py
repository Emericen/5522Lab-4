import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
torch.manual_seed(2)

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)


class MLP2(nn.Module):
	def __init__(self, input_dim = 2, hid_dim=2, output_dim=1):
		# initialze the superclass
		super(MLP2, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim)
		self.lin2 = nn.Linear(hid_dim, output_dim)
    
    def forward(self, x):
    	x = self.lin1(x)
    	x = torch.sigmoid(x)
    	x = self.lin2(x)
    	x = torch.sigmoid(x)
    	return x


def weights_init(model):
	for m in model.modules():
		if isinstance(m, nn.Linear):
			# initialize the weight tensor, here we use a normal distribution
			m.weight.data.normal_(0, 1)


model = MLP2()
weights_init(model)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)


epochs = 2001
steps = X.size(0)
for i in range(epochs):
	for j in range(steps):
		data_point = np.random.randint(X.size(0))
		x_var = Variable(X, requires_grad=False)
		y_var = Variable(Y, requires_grad=False)

		optimizer.zero_grad()
		y_hat = model(x_var)
		loss = loss_func.forward(y_hat, y_var)
		loss.backward()
		optimizer.step()

	if i % 500 == 0:
		print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))




