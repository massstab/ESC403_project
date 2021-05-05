"""
@author: dtm, yves, marszpd
"""

import pylab as py
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neighbors import KernelDensity
from matplotlib.image import imread
import numpy as np
from helpers import lv95_latlong

# =============================================================================
# Data
# =============================================================================
df = pd.read_csv('../data/tidy_data/data_merged.csv')[:1000]

features = ['Date','AccidentType','AccidentSeverityCategory',
                          'AccidentInvolvingPedestrian','AccidentInvolvingBicycle',
                          'AccidentInvolvingMotorcycle','RoadType','AccidentLocation_CHLV95_E',
                          'AccidentLocation_CHLV95_N','AvgTemperature', 'AvgRainDur',
                          'SumBikerNumber','SumPedastrianNumber', 'SumCars']

display_testing_arima = False
display_testing_slider = False
display_plot = False
display_plot_openstreet = True
# =============================================================================
# Just for testing purposes, can all be deleted!
# =============================================================================
if display_testing_arima:
    pd.set_option('display.width', 150)
    pd.set_option('display.max_columns', 10)

    # just testing. can be removed completely
    feature = features[9]
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

# =============================================================================
# this will open a browser tab
if display_testing_slider:
    x_coord = df[features[7]].values
    y_coord = df[features[8]].values
    z = df[features[2]]

    fig = px.scatter(df, x=features[7], y=features[8], animation_frame=features[0], animation_group=features[2],
            color=features[2], hover_name=features[2], range_x=[min(x_coord),max(x_coord)], range_y=[min(y_coord),max(y_coord)])

    # fig["layout"].pop("updatemenus") # optional, drop animation buttons
    plot(fig)
# =============================================================================
if display_plot:
    img = imread("../data/Zürich_map/1200px-Karte_Gemeinde_Zürich_2007.png")

    reference_coord = [2642695, 1205591]
    x_coord = df[features[7]].values.reshape((-1, 1))
    y_coord = df[features[8]].values.reshape((-1, 1))
    z = df[features[2]].values.reshape((-1, 1))

    fig = plt.figure(figsize=(19, 17), dpi=80)
    axes = fig.add_subplot(1, 1, 1)

    py.scatter(x_coord, y_coord, s=z**2, c=z, cmap='seismic')
    axes.imshow(img, extent=[min(x_coord)[0], max(x_coord)[0], min(y_coord)[0], max(y_coord)[0]])
    py.show()


# =============================================================================

if display_plot_openstreet:
    coords = np.loadtxt('../data/tidy_data/data_merged.csv', delimiter=",", skiprows=1, usecols=(7,8))
    longitude, latitude = lv95_latlong(coords[:, 0], coords[:, 1])
    BBox = (longitude.min(), longitude.max(), latitude.min(), latitude.max())

    plt.scatter(longitude, latitude, marker=".")
    plt.show()