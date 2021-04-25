import pandas as pd
import matplotlib.pyplot as plt


# Just for testing purposes, can all be deleted!

pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 10)

# just testing. can be removed completely
df = pd.read_pickle('../data/tidy_data/data_merged.pickle')
feature = 'AvgRainDur'
df_type = df[feature].resample('M').mean()
print(df_type.head(20))
x = df_type.index
y = df_type
fig, ax = plt.subplots(figsize=(10, 10))
ax.grid()
ax.bar(x, y, align='edge', width=10)
# ax.plot(x, y)
ax.scatter(x, y, color='purple')
ax.set_title(f'Feature: {feature}')
plt.show()

