import component
import pandas as pd
import numpy as np

url='https://raw.githubusercontent.com/efosler/cse5522data/master/vowelfmts.csv'
df=pd.read_csv(url)
data = np.array(df)
# print(data.shape)

# dic = {}
# for entry in data:
# 	if not entry[10] in dic:
# 		dic[entry[10]] = 0
# 	else:
# 		dic[entry[10]] += 1
# print(dic)


X = [[0,0],[0,1], [1,0], [1,1]]
Y = [[0],[1],[1],[0]]

model = component.two_layer_sigmoid()

# component.train_with_MSE(model, X, Y)
# print("//////////////////////////////")
component.train_with_CrossEntropy(model, X, Y)

