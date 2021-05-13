# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:17:45 2021
@author: dtm, marszpd
"""

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import data_all as df
import statsmodels.formula.api as sm

pd.options.display.max_columns = 5
pd.options.display.max_rows = 10
pd.set_option('display.width', 150)

features = ['Date', 'AccidentType', 'AccidentSeverityCategory', 'AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle', 'RoadType',
            'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N', 'AvgTemperature',
            'AvgRainDur', 'SumBikerNumber', 'SumPedastrianNumber', 'SumCars']

def sum_per_hour(df):

    unique_date_temp = df.groupby(['Date'])
    unique_date = unique_date_temp['Date'].count().to_frame()
    df_new = df.drop_duplicates(subset=['Date', 'AvgTemperature'])
    unique_date.insert(1, 'AvgTemperature', df_new['AvgTemperature'].to_numpy())
    # data_merged.rename(columns=new_cols, inplace=True)
    unique_date.rename(columns={'Date': 'SumAccidents'}, inplace=True)

    return unique_date

df_perhour = sum_per_hour(df)

# df_perhour.reset_index()

x = list(range(len(df_perhour.index.values)))
y = df_perhour["SumAccidents"].values
z = df_perhour["AvgTemperature"].values

# collect them in a pandas dfframe
df = pd.DataFrame({'Date': x,
                 'SumAccidents': y,
                 'AvgTemperature': z})

print(df.info())

# linear regression fit111
reg = sm.ols(formula='z ~ x + y', data=df).fit()
print(reg.summary())

from mpl_toolkits.mplot3d import Axes3D
a0, a1, a2 = reg.params
print('The fitting formula is: z = {0:} + {1:} x + {2:} y'.format(round(a0, 3),
                                                                  round(a1, 3),
                                                                  round(a2, 3)))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1)

# set up the surface plot
X = np.linspace(min(x), max(x), 100)
Y = np.linspace(min(y), max(y), 100)
XX, YY = np.meshgrid(X, Y)
ZZ = a0 + a1 * XX + a2 * YY
ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='orange')

plt.show()






