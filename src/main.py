"""
@author: dtm, yves, marszpd
"""

import pandas as pd
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt
from matplotlib.image import imread
from helpers import lv95_latlong
from datasets import data_all

# =============================================================================
# Data
# =============================================================================
df = data_all
df_count_ped_bike = pd.read_pickle('../data/tidy_data/pre_tidy_fussgaenger_velo/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.pickle')
df_count_car = pd.read_csv('../data/tidy_data/pre_tidy_auto/sid_dav_verkehrszaehlung_miv_OD2031_2012-2020.csv')

map01 = imread("../data/Zürich_map/standard.png")
map02 = imread("../data/Zürich_map/traffic.png")
map03 = imread("../data/Zürich_map/human.png")

map_calibration_angle = (8.4591, 8.6326, 47.3128, 47.4349)  # These coordinates fits the images in /data/Zürich_map

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
    long, lat = df[features[7]].values, df[features[8]].values
    longitude, latitude = lv95_latlong(long, lat)
    z = df[features[2]].values.reshape((-1, 1))

    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_xlim(map_calibration_angle[0], map_calibration_angle[1])
    ax.set_ylim(map_calibration_angle[2], map_calibration_angle[3])
    ax.set_title('Accidents Spatial Data')
    ax.scatter(longitude, latitude, s=z, c=z, cmap='seismic')
    ax.imshow(map01, extent=map_calibration_angle, aspect=('auto'))
    plt.show()


# =============================================================================
if display_plot_ped_bike:

    x_coord = df_count_ped_bike[features[7]].values.reshape((-1, 1))
    y_coord = df_count_ped_bike[features[8]].values.reshape((-1, 1))
    # z = df_count_ped_bike[features[11]].values.reshape((-1, 1))

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    map_calibration_angle = map_calibration_angle
    map01 = imread("../data/Zürich_map/map.png")

    fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
    ax.scatter(longitude, latitude, zorder=1, alpha=0.3, s=2, c="b")
    ax.set_xlim(map_calibration_angle[0], map_calibration_angle[1])
    ax.set_ylim(map_calibration_angle[2], map_calibration_angle[3])
    ax.set_title('Accidents Spatial Data')
    ax.imshow(map01, zorder=0, extent=map_calibration_angle, aspect='equal')
    plt.autoscale()
    plt.show()

# =============================================================================
if display_plot_car:

    x_coord = df_count_car[features[7]].values.reshape((-1, 1))
    y_coord = df_count_car[features[8]].values.reshape((-1, 1))
    # z = df_count_ped_bike[features[13]].values.reshape((-1, 1))

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    map01 = imread("../data/Zürich_map/map.png")


    fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
    ax.scatter(longitude, latitude, zorder=1, alpha=0.3,  s=2, c="b")
    ax.set_xlim(map_calibration_angle[0], map_calibration_angle[1])
    ax.set_ylim(map_calibration_angle[2], map_calibration_angle[3])
    ax.set_title('Accidents Spatial Data')
    ax.imshow(map01, zorder=0, extent=map_calibration_angle, aspect='equal')
    plt.autoscale()
    plt.show()

