import pandas as pd


file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(file,header=None)
df.to_csv('data.csv',index=False,header=None)