import pandas as pd
import matplotlib.pyplot as plt


# Just for testing purposes, can all be deleted!

pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 10)

# just testing. can be removed completely
df = pd.read_pickle('../data/tidy_data/data_merged.pickle')
print(df.head())
df_type = df['AvgTemperature'].resample('S').mean()
x = df_type.index.values
y = df_type
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x, y, color='purple')
plt.show()

