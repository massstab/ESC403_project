import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import (linear_model, datasets, metrics, discriminant_analysis)
from datasets import data_all

features = ['Date','AccidentType','AccidentSeverityCategory','AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle','AccidentInvolvingMotorcycle','RoadType',
            'AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N','AvgTemperature',
            'AvgRainDur','SumBikerNumber','SumPedastrianNumber', 'SumCars']

# data_all.index = pd.to_datetime(data_all.Date)
# data_all.drop(columns='Date', inplace=True)
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.set_option('display.width', 150)


severity = False
correlations = True


# =============================================================================
# Take a look at severity and vehicle involvement
# =============================================================================
if severity:
    df = data_all[['AccidentSeverityCategory', 'AccidentInvolvingPedestrian',
                   'AccidentInvolvingBicycle','AccidentInvolvingMotorcycle']]
    df = df.groupby('AccidentSeverityCategory', as_index=False).sum()
    x = df['AccidentSeverityCategory']
    y_ped = df['AccidentInvolvingPedestrian']
    y_bic = df['AccidentInvolvingBicycle']
    y_mot = df['AccidentInvolvingMotorcycle']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 10), sharex='col', sharey='col')
    ax1.scatter(x, y_ped)
    ax2.scatter(x, y_bic)
    ax3.scatter(x, y_mot)
    # ax3.set_xlabel('SeverityCategory')
    ax1.set_ylabel('Pedestrian')
    ax1.grid()
    ax2.set_ylabel('Bicycle')
    ax2.grid()
    ax3.set_ylabel('Motorcycle')
    ax3.grid()
    fig.suptitle('correlation between severity and involvement of pedestrians')
    ax3.set_xticks([1, 2, 3, 4])
    ax3.set_xticklabels(['fatalities', 'injuries', 'light injuries', 'property damage'])
    plt.show()

# =============================================================================
# Finding correlations
# =============================================================================
if correlations:
    df = data_all[['AccidentType', 'RoadType']]
    # df = df.round(0)
    # df = df.sort_values('AccidentType')
    # df0 = df[(df.AccidentType == 0)]
    # df1 = df[(df.AccidentType == 1)]
    # df2 = df[(df.AccidentType == 2)]
    # df3 = df[(df.AccidentType == 3)]
    # df4 = df[(df.AccidentType == 4)]
    # df5 = df[(df.AccidentType == 5)]
    # df6 = df[(df.AccidentType == 6)]
    # df7 = df[(df.AccidentType == 7)]
    # df8 = df[(df.AccidentType == 8)]
    # df9 = df[(df.AccidentType == 9)]

    # df_0 = df.query('AccidentType <= 7 and AccidentType >= 5 and AvgTemperature > 15')
    # print(df1.head())

    plt.scatter(df.iloc[:, 1], df.iloc[:, 0])
    # sns.pairplot(df, diag_kind="kde")
    plt.show()
