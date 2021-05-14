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

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# =============================================================================
# df = df[:1000]

features = ['Date','AccidentType','AccidentSeverityCategory','AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle','AccidentInvolvingMotorcycle','RoadType',
            'AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N','AvgTemperature',
            'AvgRainDur','SumBikerNumber','SumPedastrianNumber', 'SumCars']

map00 = imread("../data/Zürich_map/map.png")
map01 = imread("../data/Zürich_map/standard.png")
map02 = imread("../data/Zürich_map/traffic.png")
map03 = imread("../data/Zürich_map/human.png")

BBox = (8.4591, 8.6326, 47.3128, 47.4349)  # These coordinates fits the images in /data/Zürich_map

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

def visualize_kde(data, im_map, BBox, features, feature_number, feature_value,
                  x_coord_number=7, y_coord_number=8, severity_number=2,
                  visualize_seaborn=False, visualize_scipy=False,
                  visualize_sklearn=True, title=None):

    """Computes the kernel density estimation for given data, provided they lie
    on a plane with known coordinates. Different method are used:

    visualize via seaborn :
        Here the default parameters are used to get the kernel density estimation
        projected onto the x,y coordinates, as well as a projection of the x,z
        coordinates and y,z coordinates on the sides of the plot, neat!

    visualize via scipy :
        Here the kernel density estimation is made via the silverman rule of
        thumb of the bandwidth via the kwarg bw_method="silverman".

    visualize via sklearn :
        Here the kernel density estimation is made via the sklearn.neighbors
        KernelDensity class. The bandwidth is estimated via the grid search class
        in sklearn.model_selection via 2-fold crossvalidation through the use of
        the sklearn.model_selection KFold class.

    Parameters
    ----------
    data : DataFrame
        Pandas dataframe.
    im_map : ndarray
        Image displayed under the estimated kernel densities.
    BBox : list
        List of the maximum and minimum of the coordinate system.
    features : list
        Features in the DataFrame.
    feature_number : int
        Index of the feature to be used.
    feature_value : int {0,1, else int values used for the classes} or other types
        Values used to distinct the classes.
    x_coord_number : int, optional
        Index of the feature where the x coordinates are saved. The default is 7.
    y_coord_number : int, optional
        Index of the feature where the y coordinates are saved. The default is 8.
    severity_number : int or boolean False, optional
        If False marker sizes are set to 1, else the number provided is assumed
        to be the index of the accident severity category, the this information
        will be used to set the marker sizes s.t. a bigger marker indicates a worse
        accident. The default is 2.
    visualize_seaborn : boolean, optional
        Use estimation via seaborn. The default is False.
    visualize_scipy : boolean, optional
        Use estimation via scipy. The default is False.
    visualize_sklearn : boolean, optional
        Use estimation via sklearn. The default is False.

    Returns
    -------
    None.

    """

    data = df[df[features[feature_number]] == feature_value]
    print(data)
    x_coord = data[features[x_coord_number]].values.reshape((-1, 1))
    y_coord = data[features[y_coord_number]].values.reshape((-1, 1))

    longitude, latitude = lv95_latlong(x_coord, y_coord)
    X, Y = np.mgrid[BBox[0]:BBox[1]:100j, BBox[2]:BBox[3]:100j]

    if title is None:
        title = features[feature_number]

    if severity_number is not False:
        severity = data[features[severity_number]].values
        sizes = list((0.1*severity))
    else:
        sizes = list(np.ones(len(x_coord)))

    if visualize_seaborn:
        sns.jointplot(longitude[:,0], latitude[:,0], kind="kde", fill=True)
        # (sns.jointplot(longitude[:,0], latitude[:,0], color="k", marker='.').plot_joint(sns.kdeplot, n_levels=20, shade=True, alpha=0.6))


    if visualize_scipy:

        kernel = stats.gaussian_kde([longitude[:, 0], latitude[:,0]], bw_method="silverman")
        Z = np.reshape(kernel([X.ravel(), Y.ravel()]).T, X.shape)

        # plot contours of density
        fig, ax = plt.subplots(dpi=120)
        plt.title(title)
        levels = np.linspace(0, Z.max(), 20)
        plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Blues, alpha=0.5) #
        ax.scatter(longitude, latitude, color='k', s=sizes, alpha=0.5)
        ax.imshow(im_map, extent=BBox, alpha=1)
        plt.xlabel("Longitude [°]")
        plt.ylabel("Latitude [°]")
        plt.show()


    if visualize_sklearn:
        # formating data
        samples = np.vstack([longitude[:, 0], latitude[:,0]]).T
        xy_grid = np.vstack([X.ravel(), Y.ravel()]).T
        bandwidths = np.linspace(0.0, 0.1, 100)

        #setting up a grid search to find best bandwidth via 2 fold corss validation
        grid = GridSearchCV(KernelDensity(kernel='gaussian', algorithm="auto"), {'bandwidth': bandwidths},
                            cv=KFold(2))

        grid = grid.fit(samples) # perform estimation
        kde =  grid.best_estimator_ # find estimation for best suited bandwidth
        print(f"Best bandwidth: {kde.bandwidth}")

        # set up grid data
        Z = kde.score_samples(xy_grid) # score_samples returns the log-likelihood of the sample
        Z = np.exp(Z.reshape(X.shape))

        # plot contours of the density
        levels = np.linspace(0, Z.max(), 25)
        fig, ax = plt.subplots(dpi=120)
        plt.title(title)
        plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.hot_r, alpha=0.5)
        ax.imshow(im_map, alpha=1, extent=BBox, aspect='equal')
        plt.scatter(longitude, latitude, s=sizes)
        plt.xlabel("Longitude [°]")
        plt.ylabel("Latitude [°]")
        plt.show()


if __name__ == "__main__":

    titles = ["Kernel density estimation: accidents involving pedestrians, 2011-2020",
              "Kernel density estimation: accidents involving bicycles, 2011-2020",
              "Kernel density estimation: accidents involving motocycles, 2011-2020",]
    # visualize_kde(df, map00, BBox, features, 3, 1, title=titles[0])
    # visualize_kde(df, map00, BBox, features, 4, 1, title=titles[1])
    # visualize_kde(df, map00, BBox, features, 5, 1, title=titles[2])

    l = [0, 1, 2, 3, 4, 9]
    titles = ["Kernel density estimation: accidents on motorways, 2011-2020",
              "Kernel density estimation: accidents on expressways, 2011-2020",
              "Kernel density estimation: accidents on principal roads, 2011-2020",
              "Kernel density estimation: accidents on minor roads, 2011-2020",
              "Kernel density estimation: accidents on motorways side installation, 2011-2020",
              "Kernel density estimation: accidents on other road types, 2011-2020"]
    # visualize_kde(df, map00, BBox, features, 6, l[0], title=titles[0])
    # visualize_kde(df, map00, BBox, features, 6, l[1], title=titles[1])
    # visualize_kde(df, map00, BBox, features, 6, l[2], title=titles[2])
    # visualize_kde(df, map00, BBox, features, 6, l[3], title=titles[3])
    visualize_kde(df, map00, BBox, features, 6, l[4], title=titles[4])
    visualize_kde(df, map00, BBox, features, 6, l[5], title=titles[5])

    l = [1, 2, 3, 4, 5, 6, 7, 8, 9] # check ambiguity of 0 and 00 in accident type (pdf)
    titles = ["Kernel density estimation: accidents when overtaking or changing lanes, 2011-2020",
              "Kernel density estimation: accidents with rear-end collision, 2011-2020",
              "Kernel density estimation: accidents when turning left or right, 2011-2020",
              "Kernel density estimation: accidents when turning-into main road, 2011-2020",
              "Kernel density estimation: accidents when crossing the lane(s), 2011-2020",
              "Kernel density estimation: accidents with head-on collision, 2011-2020",
              "Kernel density estimation: accidents when parking, 2011-2020",
              "Kernel density estimation: accidents involving pedestrian(s), 2011-2020",
              "Kernel density estimation: accidents involving animal(s), 2011-2020"]
    visualize_kde(df, map00, BBox, features, 2, l[0], title=titles[0])
    visualize_kde(df, map00, BBox, features, 2, l[1], title=titles[1])
    visualize_kde(df, map00, BBox, features, 2, l[2], title=titles[2])
    visualize_kde(df, map00, BBox, features, 2, l[3], title=titles[3])
    visualize_kde(df, map00, BBox, features, 2, l[4], title=titles[4]) # not enough data points for this estimation
    visualize_kde(df, map00, BBox, features, 2, l[5], title=titles[5]) # not enough data points for this estimation
    visualize_kde(df, map00, BBox, features, 2, l[6], title=titles[6]) # not enough data points for this estimation
    visualize_kde(df, map00, BBox, features, 2, l[7], title=titles[7]) # not enough data points for this estimation
    visualize_kde(df, map00, BBox, features, 2, l[8], title=titles[8]) # not enough data points for this estimation
