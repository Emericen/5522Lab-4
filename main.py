import component
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

url='https://raw.githubusercontent.com/efosler/cse5522data/master/vowelfmts.csv'
df=pd.read_csv(url)

unindex_data = np.array(df)
data = np.zeros((unindex_data.shape[0], 19))

# {'ah': 174, 'ao': 257, 'ax': 114, 'eh': 235, 'ey': 414, 'ih': 282, 'iy': 467, 'ow': 916, 'uw': 369}
for row in range(len(unindex_data)):
	if unindex_data[row][-1] == 'ah':
		index = np.array([1,0,0,0,0,0,0,0,0])
	elif unindex_data[row][-1] == 'ao':
		index = np.array([0,1,0,0,0,0,0,0,0])
	elif unindex_data[row][-1] == 'ax':
		index = np.array([0,0,1,0,0,0,0,0,0])
	elif unindex_data[row][-1] == 'eh':
		index = np.array([0,0,0,1,0,0,0,0,0])
	elif unindex_data[row][-1] == 'ey':
		index = np.array([0,0,0,0,1,0,0,0,0])
	elif unindex_data[row][-1] == 'ih':
		index = np.array([0,0,0,0,0,1,0,0,0])
	elif unindex_data[row][-1] == 'iy':
		index = np.array([0,0,0,0,0,0,1,0,0])
	elif unindex_data[row][-1] == 'ow':
		index = np.array([0,0,0,0,0,0,0,1,0])
	elif unindex_data[row][-1] == 'uw':
		index = np.array([0,0,0,0,0,0,0,0,1])
	data[row] = np.append(unindex_data[row][:-1], index)

X = data[:,:2]
Y = data[:,-9:]

print(X[0])
print(Y[0])

print("input size:" + str(len(X[0])))
print("output size:" + str(len(Y[0])))


model = component.two_layer_sigmoid(input_dim=2, hid_dim=100, output_dim=9)
# component.train_with_CrossEntropy(model, X, Y, epochs=10)


