import pandas as pd
pd.set_option('display.max_columns', 42)

data = pd.read_csv('../datasets/2015-building-energy-benchmarking.csv')

data['DataYear'] = data['DataYear'].astype(object)
print(data.dtypes)
