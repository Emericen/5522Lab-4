import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(2)

X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
Y = torch.Tensor([0,1,1,0]).view(-1,1)


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
				m.weight.data.fill_(1)
				m.bias.data.fill_(0)

	def forward(self, x):
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
				m.weight.data.fill_(-1)
				m.bias.data.fill_(0)

	def forward(self, x):
		x = self.lin1(x)
		x = torch.relu(x)
		x = self.lin2(x)
		x = torch.relu(x)
		return x


class three_layer_sigmoid(nn.Module):
	def __init__(self, input_dim=2, hid_dim_1=2, hid_dim_2=2, output_dim=1):
		super(two_layer_relu, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim_1)
		self.lin2 = nn.Linear(hid_dim_1, hid_dim_2)
		self.lin3 = nn.Linear(hid_dim_2, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.fill_(-1)
				m.bias.data.fill_(0)

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
		super(two_layer_relu, self).__init__()
		self.lin1 = nn.Linear(input_dim, hid_dim_1)
		self.lin2 = nn.Linear(hid_dim_1, hid_dim_2)
		self.lin3 = nn.Linear(hid_dim_2, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.fill_(-1)
				m.bias.data.fill_(0)

	def forward(self, x):
		x = self.lin1(x)
		x = torch.relu(x)
		x = self.lin2(x)
		x = torch.relu(x)
		x = self.lin3(x)
		x = torch.relu(x)
		return x


class MLP2(nn.Module):
	def __init__(self, input_dim = 2, hid_dim=2, output_dim=1):
		
		super(MLP2, self).__init__()

		self.lin1 = nn.Linear(input_dim, hid_dim)
		self.lin2 = nn.Linear(hid_dim, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				# m.weight.data.normal_(0, 1)
				m.weight.data.fill_(1)

	def forward(self, x):
		x = self.lin1(x)
		x = torch.sigmoid(x)
		x = self.lin2(x)
		x = torch.sigmoid(x)
		return x

	def train(self, X, Y, epochs=2001, lr=0.02, momentum=0.9):
		loss_func = nn.MSELoss()
		optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
		steps = X.size(0)
		for i in range(epochs):
			for j in range(steps):
				data_point = np.random.randint(X.size(0))
				x_var = Variable(X, requires_grad=False)
				y_var = Variable(Y, requires_grad=False)

				optimizer.zero_grad()
				y_hat = self(x_var)
				loss = loss_func.forward(y_hat, y_var)
				loss.backward()
				optimizer.step()

			if i % 500 == 0:
				print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()))

		print("Complete!")

# print(X)
# print("//////////////////////")
# print(Y)

model = two_layer_relu()
A = torch.Tensor([[-1,2]])
B = model.forward(A)
print("result:", B)



# model = MLP2()
# model.train(X, Y)

# model_params = list(model.parameters())
# model_1_w = model_params[0].data.numpy()
# model_1_b = model_params[1].data.numpy()
# model_2_w = model_params[2].data.numpy()
# model_2_b = model_params[3].data.numpy()

# print("//////////////////////")
# print(model_1_w)
# print(model_1_b)
# print("//////////////////////")
# print(model_2_w)
# print(model_2_b)


