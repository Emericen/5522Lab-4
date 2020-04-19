import component
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

url='https://raw.githubusercontent.com/efosler/cse5522data/master/vowelfmts.csv'
df=pd.read_csv(url)
data = np.array(df)


# for entry in data:
# 	if entry[-1]


print(data)


# train_data, test_data, train_targets, test_targets = train_test_split(df[['f1','f2']],df[['close','near-close+','close-mid+','mid+','open-mid+','back','central-or-back','rounded','vowel']])

# train_data = np.array(train_data)
# train_targets = np.array(train_targets)

# test_data = np.array(test_data)
# test_targets = np.array(test_targets)


dic = {}
for entry in data:
	if not entry[10] in dic:
		dic[entry[10]] = 0
	else:
		dic[entry[10]] += 1
print(dic)


# X = torch.Tensor(X)
# Y = torch.Tensor(Y)


# model = component.two_layer_sigmoid(input_dim=2, hid_dim=100, output_dim=9)
# component.train_with_CrossEntropy(model, X, Y)








