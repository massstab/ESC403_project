import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/raw/RoadTrafficAccidentLocations.csv")
print(df.keys())
df = df['AccidentYear']
df = df.value_counts()
y = df.values
x = df.keys()
# print(df.values)
# print(df.keys)
plt.grid()
plt.bar(x, y)
plt.show()
