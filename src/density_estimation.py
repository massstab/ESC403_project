# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:01:31 2021
@author: marszpd
"""

import numpy as np
import pandas as pd
# import plotly.express as px
# from plotly.offline import plot
import matplotlib.pyplot as plt
import scipy.stats as stats
# from statsmodels.tsa.arima.model import ARIMA
from matplotlib.image import imread
from helpers import lv95_latlong
import seaborn as sns

from datasets import data_all as df

# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold

# df_count_ped_bike = pd.read_pickle('../data/tidy_data/pre_tidy_fussgaenger_velo/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.pickle')
# df_count_car = pd.read_csv('../data/tidy_data/pre_tidy_auto/sid_dav_verkehrszaehlung_miv_OD2031_2012-2020.csv')

map01 = imread("../data/Zürich_map/standard.png")
map02 = imread("../data/Zürich_map/traffic.png")
map03 = imread("../data/Zürich_map/human.png")

BBox = (8.4591, 8.6326, 47.3128, 47.4349)  # These coordinates fits the images in /data/Zürich_map

features = ['Date','AccidentType','AccidentSeverityCategory','AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle','AccidentInvolvingMotorcycle','RoadType',
            'AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N','AvgTemperature',
            'AvgRainDur','SumBikerNumber','SumPedastrianNumber', 'SumCars']


x_coord = df[features[7]].values.reshape((-1, 1))
y_coord = df[features[8]].values.reshape((-1, 1))
# z = df_count_ped_bike[features[13]].values.reshape((-1, 1))

longitude, latitude = lv95_latlong(x_coord, y_coord)
map01 = imread("../data/Zürich_map/map.png")

display_ped = True
display_bike = False
display_motor = False
# =============================================================================
# Display whole data st
# =============================================================================
# fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
# ax.scatter(longitude, latitude, zorder=1, alpha=0.3,  s=2, c="b")
# ax.set_xlim(BBox[0], BBox[1])
# ax.set_ylim(BBox[2], BBox[3])
# ax.set_title('Accidents Spatial Data')
# ax.imshow(map01, zorder=0, extent=BBox, aspect='equal')
# plt.autoscale()
# plt.show()

# sns.jointplot(longitude[:,0], latitude[:,0])
# =============================================================================
# Display KerneldensityEstimation
# =============================================================================

if display_ped:
    data_ped = df[df[features[3]] == 1]
    x_coord = data_ped[features[7]].values.reshape((-1, 1))
    y_coord = data_ped[features[8]].values.reshape((-1, 1))
    severity = data_ped[features[2]].values
    sizes = list((severity)**3)

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    X, Y = np.mgrid[BBox[0]:BBox[1]:100j, BBox[2]:BBox[3]:100j]

    kernel = stats.gaussian_kde([longitude[:, 0], latitude[:,0]], "silverman")
    Z = np.reshape(kernel([X.ravel(), Y.ravel()]).T, X.shape)

    fig, ax = plt.subplots(dpi=120)
    # ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=BBox)
    levels = np.linspace(0, Z.max(), 20)
    plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Blues) #
    ax.scatter(longitude, latitude, color='k', s=sizes, alpha=0.1)
    sns.jointplot(longitude[:,0], latitude[:,0], kind="kde", fill=True)
    # (sns.jointplot(longitude[:,0], latitude[:,0], color="k", marker='.').plot_joint(sns.kdeplot, n_levels=20, shade=True, alpha=0.6))
    plt.show()


if display_bike:
    data_bike = df[df[features[4]] == 1]
    x_coord = data_bike[features[7]].values.reshape((-1, 1))
    y_coord = data_bike[features[8]].values.reshape((-1, 1))
    severity = data_bike[features[2]].values
    sizes = list((severity)**3)

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    X, Y = np.mgrid[BBox[0]:BBox[1]:100j, BBox[2]:BBox[3]:100j]

    kernel = stats.gaussian_kde([longitude[:, 0], latitude[:,0]], "silverman")
    Z = np.reshape(kernel([X.ravel(), Y.ravel()]).T, X.shape)

    fig, ax = plt.subplots(dpi=120)
    # ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=BBox)
    levels = np.linspace(0, Z.max(), 20)
    plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Blues) #
    ax.scatter(longitude, latitude, color='k', s=sizes, alpha=0.1)
    sns.jointplot(longitude[:,0], latitude[:,0], kind="kde", fill=True)
    # (sns.jointplot(longitude[:,0], latitude[:,0], color="k", marker='.').plot_joint(sns.kdeplot, n_levels=20, shade=True, alpha=0.6))
    plt.show()


if display_motor:
    data_motor = df[df[features[5]] == 1]
    x_coord = data_motor[features[7]].values.reshape((-1, 1))
    y_coord = data_motor[features[8]].values.reshape((-1, 1))
    severity = data_motor[features[2]].values
    sizes = list((severity)**3)

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    X, Y = np.mgrid[BBox[0]:BBox[1]:100j, BBox[2]:BBox[3]:100j]

    kernel = stats.gaussian_kde([longitude[:, 0], latitude[:,0]], "silverman")
    Z = np.reshape(kernel([X.ravel(), Y.ravel()]).T, X.shape)

    fig, ax = plt.subplots(dpi=120)
    # ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=BBox)
    levels = np.linspace(0, Z.max(), 20)
    plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Blues) #
    ax.scatter(longitude, latitude, color='k', s=sizes, alpha=0.1)
    sns.jointplot(longitude[:,0], latitude[:,0], kind="kde", fill=True)
    # (sns.jointplot(longitude[:,0], latitude[:,0], color="k", marker='.').plot_joint(sns.kdeplot, n_levels=20, shade=True, alpha=0.6))
    plt.show()

# =============================================================================
# Second approach
# =============================================================================
# via KernelDensity
# bandwidths = 10 ** np.linspace(-1., 1., 100)
# grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#                     {'bandwidth': bandwidths},
#                     cv=KFold(2))

# kde = grid.fit(samples)
# kde.fit(samples.T)
# print(kde.best_estimator_)

# kde = KernelDensity(bandwidth=0.1, kernel='gaussian', algorithm='ball_tree')
# print(samples.shape)
# kde.fit(samples.T)
# print(positions.shape)

# Z = kde.score_samples(positions.T)
# Z = Z.reshape(X.shape)
# # plot contours of the density
# levels = np.linspace(0, Z.max(), 25)
# plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
# plt.scatter(longitude, latitude, s=2)
# plt.show()
