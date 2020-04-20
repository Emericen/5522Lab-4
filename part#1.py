import component
import pandas as pd
import torch

url = 'https://raw.githubusercontent.com/efosler/cse5522data/master/vowelfmts.csv'
df = pd.read_csv(url)
data = df.values.tolist()
# print(data[0])

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

# X = [[entry[0],entry[1]] for entry in data]
X = [[entry[i] for i in range(10)] for entry in data]
Y = [entry[-1] for entry in data]

model = component.two_layer_sigmoid(input_dim=10, hid_dim=100, output_dim=9)
component.train(model, X, Y, epochs	=20)
component.test(model, X, Y)
