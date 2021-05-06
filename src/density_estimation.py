# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:01:31 2021
@author: marszpd
"""

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.image import imread
from helpers import lv95_latlong
import seaborn as sns
import KernelDensity


df = pd.read_csv('../data/tidy_data/data_merged.csv')[:1000]
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

# =============================================================================
# Display whole data st
# =============================================================================
fig, ax = plt.subplots(figsize=(15, 12), dpi=80)
ax.scatter(longitude, latitude, zorder=1, alpha=0.3,  s=2, c="b")
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
ax.set_title('Accidents Spatial Data')
ax.imshow(map01, zorder=0, extent=BBox, aspect='equal')
plt.autoscale()
plt.show()

# sns.jointplot(longitude[:,0], latitude[:,0])
# =============================================================================
# Display KerneldensityEstimation
# =============================================================================

samples = np.vstack([longitude[:,0], latitude[:,0]]) # format data
X, Y = np.mgrid[BBox[0]:BBox[1]:1000j, BBox[2]:BBox[3]:1000j]
positions = np.vstack([X.ravel(), Y.ravel()])

# # via scipy.stats
# kernel = stats.gaussian_kde(samples)
# Z = np.reshape(kernel(positions).T, X.shape)

# fig, ax = plt.subplots()
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=BBox)
# ax.plot(longitude, latitude, 'k.', markersize=2)
# plt.show()

# via KernelDensity
kde = KernelDensity(bandwidth=0.04, metric='haversine', kernel='gaussian', algorithm='ball_tree')
kde.fit(samples)

Z = kde.score_samples(positions)
# plot contours of the density
levels = np.linspace(0, Z.max(), 25)
plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
