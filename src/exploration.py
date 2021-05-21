import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import (linear_model, datasets, metrics, discriminant_analysis)
from datasets import data_all as df

features = ['Date', 'AccidentType', 'AccidentSeverityCategory', 'AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle', 'RoadType',
            'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N', 'AvgTemperature',
            'AvgRainDur', 'SumBikerNumber', 'SumPedestrianNumber', 'SumCars']

df.index = pd.to_datetime(df.Date)
df.drop(columns='Date', inplace=True)
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.set_option('display.width', 150)


severity = False
correlations = True


# =============================================================================
# Take a look at severity and vehicle involvement
# =============================================================================
if severity:
    df = df[['AccidentSeverityCategory', 'AccidentInvolvingPedestrian',
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
    # df = data_all[['AvgTemperature', 'AccidentSeverityCategory']]
    # df = df.round(0)
    # df = df.sort_values('AccidentType')
    # df0 = df[(df.AccidentType == 0)]
    # df_0 = df.query('AccidentType <= 7 and AccidentType >= 5 and AvgTemperature > 15')

    # df = df[(df['RoadType'] != 2) & (df['RoadType'] != 3)]
    df = df[(df['AccidentType'] == 0)]
    # df.dropna('AccidentType', inplace=True)

    def find_corr(featurex, featurey, category):
        fig, ax = plt.subplots()
        scatter = ax.scatter(df[featurex], df[featurey], c=df[category], s=2, cmap='tab10')
        legend = ax.legend(*scatter.legend_elements(), loc="upper right", title=category)
        ax.add_artist(legend)
        ax.set_xlabel(featurex)
        ax.set_ylabel(featurey)
        sns.pairplot(df, diag_kind="kde")
        plt.show()

    find_corr('SumCars', 'AvgRainDur', 'RoadType')
