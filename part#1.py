import component
import pandas as pd
import numpy as np

url='https://raw.githubusercontent.com/efosler/cse5522data/master/vowelfmts.csv'
df=pd.read_csv(url)
data = np.array(df)

model = component.two_layer_sigmoid(input_dim=10, hidd_dim=100, output_dim=9)


















