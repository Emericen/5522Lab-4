import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(2)


class two_layer_flat(nn.Module):
	def __init__(self, input_dim=2, hid_dim=2, output_dim=1):
		super(two_layer_flat, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim)
		self.lin2 = nn.Linear(hid_dim, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.fill_(1)
				m.bias.data.fill_(0)

	def forward(self, x):
		x = self.lin1(x)
		x = self.lin2(x)
		return x


class two_layer_sigmoid(nn.Module):
	def __init__(self, input_dim=2, hid_dim=2, output_dim=1):
		super(two_layer_sigmoid, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim)
		self.lin2 = nn.Linear(hid_dim, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0,1)

	def forward(self, x):
		x = torch.Tensor(x)
		x = self.lin1(x)
		x = torch.sigmoid(x)
		x = self.lin2(x)
		x = torch.sigmoid(x)
		return x


class two_layer_relu(nn.Module):
	def __init__(self, input_dim=2, hid_dim=2, output_dim=1):
		super(two_layer_relu, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim)
		self.lin2 = nn.Linear(hid_dim, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0,1)

	def forward(self, x):
		x = self.lin1(x)
		x = torch.relu(x)
		x = self.lin2(x)
		x = torch.relu(x)
		return x


class three_layer_sigmoid(nn.Module):
	def __init__(self, input_dim=2, hid_dim_1=2, hid_dim_2=2, output_dim=1):
		super(three_layer_sigmoid, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim_1)
		self.lin2 = nn.Linear(hid_dim_1, hid_dim_2)
		self.lin3 = nn.Linear(hid_dim_2, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0,1)

	def forward(self, x):
		x = self.lin1(x)
		x = torch.sigmoid(x)
		x = self.lin2(x)
		x = torch.sigmoid(x)
		x = self.lin3(x)
		x = torch.sigmoid(x)
		return x


class three_layer_relu(nn.Module):
	def __init__(self, input_dim=2, hid_dim_1=2, hid_dim_2=2, output_dim=1):
		super(three_layer_relu, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim_1)
		self.lin2 = nn.Linear(hid_dim_1, hid_dim_2)
		self.lin3 = nn.Linear(hid_dim_2, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0,1)

	def forward(self, x):
		x = self.lin1(x)
		x = torch.relu(x)
		x = self.lin2(x)
		x = torch.relu(x)
		x = self.lin3(x)
		x = torch.relu(x)
		return x	

def train(model, x, y, epochs=5, lr=1, momentum=0.9):
	X = torch.Tensor(x)
	Y = torch.Tensor(y).type(torch.LongTensor)
	loss_func = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	steps = X.size(0)
	for i in range(epochs):
		for j in range(steps):
			data_point = np.random.randint(X.size(0))
			x_var = Variable(X, requires_grad=False)
			y_var = Variable(Y, requires_grad=False)

			optimizer.zero_grad()
			y_hat = model(x_var)

			loss = loss_func(y_hat, y_var)
			loss.backward()
			optimizer.step()	
		print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))


def test(model, X, Y):
	correct = 0
	for index in range(len(X)):
		y = model(X[index])
		result = [0 if not i == max(y) else 1 for i in y]
		if result.index(1) == Y[index]:
			correct += 1

	print("corrected {0}/{1}, a ratio of {2}.".format(correct, len(X), correct/len(X)))


# A = torch.Tensor([1,2,3])
# print(F.softmax(A, dim=0))

