import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# Just for testing purposes, can all be deleted!

pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 10)

# just testing. can be removed completely
df = pd.read_pickle('../data/tidy_data/data_merged.pickle')
feature = 'AvgTemperature'
df_type = df[feature].resample('D').mean()
print(df_type.head(20))
x = df_type.index
y = df_type
model = ARIMA(y)
model_fit = model.fit()
print(model_fit.summary())
fig, ax = plt.subplots(figsize=(16, 10))
ax.grid()
# ax.bar(x, y, align='edge', width=10)
ax.plot(x, y)
# ax.scatter(x, y, color='purple')
ax.set_title(f'Feature: {feature}')
plt.show()

