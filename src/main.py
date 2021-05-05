"""
@author: dtm, yves, marszpd
"""

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
# df_count_ped_bike = pd.read_pickle('../data/tidy_data/pre_tidy_fussgaenger_velo/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.pickle')
# df_count_car = pd.read_csv('../data/tidy_data/pre_tidy_auto/sid_dav_verkehrszaehlung_miv_OD2031_2012-2020.csv')

std_map = imread("../data/Zürich_map/standard.png")
BBox = (8.4591, 8.6326, 47.3128, 47.4349)  # These coordinates fits the images in /data/Zürich_map

map_calibration_meters = (2677538.0, 2689354.0, 1241844.0, 1254133.0)
map_calibration_angle = (8.4656, 8.6214, 47.3226, 47.4327)

features = ['Date','AccidentType','AccidentSeverityCategory',
                          'AccidentInvolvingPedestrian','AccidentInvolvingBicycle',
                          'AccidentInvolvingMotorcycle','RoadType','AccidentLocation_CHLV95_E',
                          'AccidentLocation_CHLV95_N','AvgTemperature', 'AvgRainDur',
                          'SumBikerNumber','SumPedastrianNumber', 'SumCars']

display_testing_arima = False
display_testing_slider = False
display_plot = True
display_plot_ped_bike = False
display_plot_car = False

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
    longitude, latitude = lv95_latlong(df[features[7]].values, df[features[8]].values)
    z = df[features[2]].values.reshape((-1, 1))

    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_title('Accidents Spatial Data')
    ax.scatter(longitude, latitude, s=z**2, c=z, cmap='seismic')
    ax.imshow(std_map, extent=BBox, aspect=('auto'))
    ax.show()


# =============================================================================
if display_plot_ped_bike:

    x_coord = df_count_ped_bike[features[7]].values.reshape((-1, 1))
    y_coord = df_count_ped_bike[features[8]].values.reshape((-1, 1))
    # z = df_count_ped_bike[features[11]].values.reshape((-1, 1))

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    BBox = map_calibration_angle
    std_map = imread("../data/Zürich_map/map.png")

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
    std_map = imread("../data/Zürich_map/map.png")


    fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
    ax.scatter(longitude, latitude, zorder=1, alpha=0.3,  s=2, c="b")
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.set_title('Accidents Spatial Data')
    ax.imshow(std_map, zorder=0, extent=map_calibration_angle, aspect='equal')
    plt.autoscale()
    plt.show()

