import component
import pandas as pd
import numpy as np

url='https://raw.githubusercontent.com/efosler/cse5522data/master/vowelfmts.csv'
df=pd.read_csv(url)
data = np.array(df)


model_1 = component.two_layer_sigmoid(input_dim=10, hid_dim=100, output_dim=9)
model_2 = component.two_layer_relu(input_dim=10, hid_dim=100, output_dim=9)

model_3 = component.three_layer_sigmoid(input_dim=10, hid_dim_1=100, hid_dim_2=100, output_dim=9)
model_4 = component.three_layer_relu(input_dim=10, hid_dim_1=100, hid_dim_2=100, output_dim=9)






















