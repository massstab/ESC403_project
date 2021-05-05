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
df = pd.read_csv('../data/tidy_data/data_merged.csv')
df_count_ped_bike = pd.read_pickle('../data/tidy_data/pre_tidy_fussgaenger_velo/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.pickle')
df_count_car = pd.read_csv('../data/tidy_data/pre_tidy_auto/sid_dav_verkehrszaehlung_miv_OD2031_2012-2020.csv')

map_calibration_meters = (2677538.0, 2689354.0, 1241844.0, 1254133.0)
map_calibration_angle = (8.4656, 8.6214, 47.3226, 47.4327)

features = ['Date','AccidentType','AccidentSeverityCategory',
                          'AccidentInvolvingPedestrian','AccidentInvolvingBicycle',
                          'AccidentInvolvingMotorcycle','RoadType','AccidentLocation_CHLV95_E',
                          'AccidentLocation_CHLV95_N','AvgTemperature', 'AvgRainDur',
                          'SumBikerNumber','SumPedastrianNumber', 'SumCars']

display_testing_arima = False
display_testing_slider = False
display_plot = False
display_plot_openstreet = True
display_plot_ped_bike = True
display_plot_car = False
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
    img = imread("../data/Zürich_map/map_big.png")

    x_coord = df[features[7]].values.reshape((-1, 1))
    y_coord = df[features[8]].values.reshape((-1, 1))
    z = df[features[2]].values.reshape((-1, 1))

    fig = plt.figure(figsize=(19, 17), dpi=80)
    axes = fig.add_subplot(1, 1, 1)

    py.scatter(x_coord, y_coord, s=z**2, c=z, cmap='seismic')
    axes.imshow(img, extent=map_calibration_meters)
    py.show()

# =============================================================================
if display_plot_openstreet:
    coords = np.loadtxt('../data/tidy_data/data_merged.csv', delimiter=",", skiprows=1, usecols=(7,8))
    longitude, latitude = lv95_latlong(coords[:, 0], coords[:, 1])
    BBox = (longitude.min(), longitude.max(), latitude.min(), latitude.max()) # map_big has been constructed to fit this parameters
    std_map = imread("../data/Zürich_map/map_big.png")
    z = df[features[3]].values.reshape((-1, 1))


    fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
    ax.scatter(longitude, latitude, zorder=1, alpha=0.3, s=(z/max(z))**4, c=z, cmap='hsv')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_title('Accidents Spatial Data')
    ax.imshow(std_map, zorder=0, extent=map_calibration_angle, aspect='equal')
    plt.autoscale()
    # plt.show()

# =============================================================================
if display_plot_ped_bike:

    x_coord = df_count_ped_bike[features[7]].values.reshape((-1, 1))
    y_coord = df_count_ped_bike[features[8]].values.reshape((-1, 1))
    # z = df_count_ped_bike[features[11]].values.reshape((-1, 1))

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    BBox = map_calibration_angle
    std_map = imread("../data/Zürich_map/map_big.png")

    # fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
    ax.scatter(longitude, latitude, zorder=1, alpha=0.3, s=2, c="b")
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_title('Accidents Spatial Data')
    ax.imshow(std_map, zorder=0, extent=map_calibration_angle, aspect='equal')
    plt.autoscale()
    plt.show()

# =============================================================================
if display_plot_car:

    x_coord = df_count_car[features[7]].values.reshape((-1, 1))
    y_coord = df_count_car[features[8]].values.reshape((-1, 1))
    # z = df_count_ped_bike[features[13]].values.reshape((-1, 1))

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    BBox = map_calibration_angle
    std_map = imread("../data/Zürich_map/map_big.png")


    fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
    ax.scatter(longitude, latitude, zorder=1, alpha=0.3,  s=2, c="b")
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_title('Accidents Spatial Data')
    ax.imshow(std_map, zorder=0, extent=map_calibration_angle, aspect='equal')
    plt.autoscale()
    plt.show()

