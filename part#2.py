import component
import pandas as pd

url = 'https://raw.githubusercontent.com/efosler/cse5522data/master/vowelfmts.csv'
df = pd.read_csv(url)
data = df.values.tolist()

# {'ah': 174, 'ao': 257, 'ax': 114, 'eh': 235, 'ey': 414, 'ih': 282, 'iy': 467, 'ow': 916, 'uw': 369}
for row in range(len(data)):
	if data[row][-1] == 'ah':
		data[row][-1] = 0
	elif data[row][-1] == 'ao':
		data[row][-1] = 1
	elif data[row][-1] == 'ax':
		data[row][-1] = 2
	elif data[row][-1] == 'eh':
		data[row][-1] = 3
	elif data[row][-1] == 'ey':
		data[row][-1] = 4
	elif data[row][-1] == 'ih':
		data[row][-1] = 5
	elif data[row][-1] == 'iy':
		data[row][-1] = 6
	elif data[row][-1] == 'ow':
		data[row][-1] = 7
	elif data[row][-1] == 'uw':
		data[row][-1] = 8

X = [[entry[0],entry[1]] for entry in data]
Y = [entry[-1] for entry in data]


epochs = 5
lr = 0.1
momentum = 0.9

model_1 = component.two_layer_sigmoid(input_dim=2, hid_dim=50, output_dim=9)
model_2 = component.three_layer_sigmoid(input_dim=2, hid_dim_1=50, hid_dim_2=50, output_dim=9)

model_3 = component.two_layer_sigmoid(input_dim=2, hid_dim=150, output_dim=9)
model_4 = component.three_layer_sigmoid(input_dim=2, hid_dim_1=150, hid_dim_2=150, output_dim=9)

model_5 = component.two_layer_relu(input_dim=2, hid_dim=50, output_dim=9)
model_6 = component.three_layer_relu(input_dim=2, hid_dim_1=50, hid_dim_2=50, output_dim=9)

model_7 = component.two_layer_relu(input_dim=2, hid_dim=150, output_dim=9)
model_8 = component.three_layer_relu(input_dim=2, hid_dim_1=150, hid_dim_2=150, output_dim=9)

print("Training sigmoid w/ size [2, 50, 9]:")
component.train(model_1, X, Y, epochs=epochs, lr=lr, momentum=momentum)
print("Training sigmoid w/ size [2, 50, 50, 9]:")
component.train(model_2, X, Y, epochs=epochs, lr=lr, momentum=momentum)

print("Training sigmoid w/ size [2, 150, 9]:")
component.train(model_3, X, Y, epochs=epochs, lr=lr, momentum=momentum)
print("Training sigmoid w/ size [2, 150, 150, 9]:")
component.train(model_4, X, Y, epochs=epochs, lr=lr, momentum=momentum)

print("Training relu w/ size [2, 50, 9]:")
component.train(model_5, X, Y, epochs=epochs, lr=lr, momentum=momentum)
print("Training relu w/ size [2, 50, 50, 9]:")
component.train(model_6, X, Y, epochs=epochs, lr=lr, momentum=momentum)

print("Training relu w/ size [2, 150, 9]:")
component.train(model_7, X, Y, epochs=epochs, lr=lr, momentum=momentum)
print("Training relu w/ size [2, 150, 150, 9]:")
component.train(model_8, X, Y, epochs=epochs, lr=lr, momentum=momentum)


