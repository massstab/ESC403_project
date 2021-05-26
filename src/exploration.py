import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import data_all as df
from helpers import prepare_data_classification

features = ['Date', 'AccidentType', 'AccidentSeverityCategory', 'AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle', 'RoadType',
            'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N', 'AvgTemperature',
            'AvgRainDur', 'SumBikerNumber', 'SumPedestrianNumber', 'SumCars']

df.index = pd.to_datetime(df.Date)
df.drop(columns='Date', inplace=True)
pd.options.display.max_columns = 100
pd.set_option('display.width', 200)


severity = False
correlations = False
sb_pairplot = False
stations_barplot = True


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
    # print(df.head(20))
    # df = df[(df['AccidentSeverityCategory'] == 1)]
    # print(df.info())
    # df.dropna('AccidentType', inplace=True)

    def find_corr(featurex, featurey, category, featurez=None, axes3d=False):
        if axes3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(df[featurex], df[featurey], df[featurez], c=df[category], s=2, cmap='tab10')
            legend = ax.legend(*scatter.legend_elements(), loc="upper right", title=category)
            ax.add_artist(legend)
            ax.set_xlabel(featurex)
            ax.set_ylabel(featurey)
            ax.set_zlabel(featurez)
        else:
            fig, ax = plt.subplots()
            scatter = ax.scatter(df[featurex], df[featurey], c=df[category], s=2, cmap='tab10')
            legend = ax.legend(*scatter.legend_elements(), loc="upper right", title=category)
            ax.add_artist(legend)
            ax.set_xlabel(featurex)
            ax.set_ylabel(featurey)
        plt.show()


    # find_corr('AccidentLocation_CHLV95_N', 'AccidentLocation_CHLV95_E', 'AccidentType', 'RoadType', axes3d=True)
    # find_corr('AccidentLocation_CHLV95_N', 'AccidentLocation_CHLV95_E', 'AccidentType', 'RoadType')

if sb_pairplot:
    df = df.sample(10000)
    df = prepare_data_classification(df, drop_some=False)
    df = df[df['RainDur'] <= 60]
    features = ['SumBikerNumber', 'SumCars', 'SumPedestrianNumber']
    df['weekday'] = df.index.day_name()
    df = df.groupby('weekday')[features].sum()
    print(df.head(20))
    # target = 'Severity'
    # num_units = df[target].nunique()
    # pal = sns.color_palette(n_colors=num_units)
    # pp = sns.pairplot(df, vars=features, hue=target, diag_kind="auto", aspect=1.44, palette=pal, kind='scatter', corner=False, plot_kws={'s': 20})
    # plt.savefig(f'../presentation/figures/pairplot_{target}_2.jpg', dpi=90)
    # plt.show()

if stations_barplot:
    from datasets import data_m

    print(data_m.head())