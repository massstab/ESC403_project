import pandas as pd
import matplotlib.pyplot as plt

# just testing. can be removed completely
df = pd.read_pickle('../data/tidy_data/data_merged.pickle')
df = df
x = df.index.year
y = df['AvgTemperature']
print(x)
print(y)
print(df.head)
plt.grid()
plt.bar(x, y)
plt.show()
