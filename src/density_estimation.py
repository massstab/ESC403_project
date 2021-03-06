# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:01:31 2021
@author: marszpd
"""

import matplotlib
import numpy as np
import pandas as pd
# import plotly.express as px
# from plotly.offline import plot
import matplotlib.pyplot as plt
import scipy.stats as stats
# from statsmodels.tsa.arima.model import ARIMA
from matplotlib.image import imread
from helpers import lv95_latlong, ccolormap
import seaborn as sns

from datasets import data_all as df

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# just for marcelo
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\FFmpeg\\bin\\ffmpeg.exe'

# =============================================================================
# df = df[:1000]

features = ['Date','AccidentType','AccidentSeverityCategory','AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle','AccidentInvolvingMotorcycle','RoadType',
            'AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N','AvgTemperature',
            'AvgRainDur','SumBikerNumber','SumPedestrianNumber', 'SumCars']

map00 = imread("../data/zurich_map/map.png")
map01 = imread("../data/zurich_map/standard.png")
map02 = imread("../data/zurich_map/traffic.png")
map03 = imread("../data/zurich_map/human.png")

BBox = (8.4591, 8.6326, 47.3128, 47.4349)  # These coordinates fits the images in /data/zurich_map

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
# caution no error handeling implemented
def visualize_kde(df, im_map, BBox, features, feature_number, feature_value, date_number=0,
                  x_coord_number=7, y_coord_number=8, severity_number=2, visualize_real_data=True,
                  visualize_seaborn=False, visualize_scipy=False,
                  visualize_sklearn=True, title=None, whole_data=True,
                  day_time=(False, 4), seasons=False, temperature=(False, 5),
                  rain_dur=(False, 15), sumped=(False, 10), sumbyc=(False, 10),
                  sumcars=(False, 10), animation=False, animation_save_dir=False,
                  interpolate=False, interpol_nframes=3):

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

    Animation :
        If one chooses one of the options for which multiple plots get generated
        one can choose the argument animation (+ setting animation_save_dir ) to
        create an animation of the evolving kernel density.

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
    feature_number : int or "all"
        Index of the feature to be used. If "all" the whole data set withot feature
        dicrimination gets used.
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
    visualize_real_data : boolean, optional
        If True the real data get visualized with a scatter plot. The default is True.
    visualize_seaborn : boolean, optional
        Use estimation via seaborn. The default is False.
    visualize_scipy : boolean, optional
        Use estimation via scipy. The default is False.
    visualize_sklearn : boolean, optional
        Use estimation via sklearn. The default is False.
    whole_data : boolean, optional
        Use whole_data to get a single kernel density estimation of the whole data
        set with the set specifications. The default is True.
    day_time : tuple (boolean, "day_parts": int), optional
        If the boolean in day_time is set to True, the integer in the tuple day_time
        will be used to devide the day into the value given by "day_parts.
        The default is (False, 4).
    seasons : boolean, optional
        If True the kde gets computed for the 4 seasons. The default is False.
    temperature : tuple  (boolean, "temperature_step"), optional
        If the boolean in temperature is set to True, the kde will be computed in
        the given temperature steps (in Kelvin [K]). The default is (False, 5), (5 K)
    rain_dur : tuple (boolean, "rain_duration_step"), optional
        If the boolean in rain_dur is True the kde will be computed for the different
        "rain_durations_steps" (in minutes [min]). The default is (False, 15). (15 min)
    sumped : tuple (boolean, "number of pedestrians step"), optional
        If the boolean in sumped is True the kde will be computed for the different
        "number of pedestrians step". The default is (False, 10). (10 pedestrains)
    sumbyc : tuple (boolean, "number of bycicles step"), optional
        If the boolean in sumbyc is True the kde will be computed for the different
        "number of bysicles step". The default is (False, 10). (10 bysicles)
    sumcars : tuple (boolean, "number of cars step"), optional
        If the boolean in sumcars is True the kde will be computed for the different
        "number of cars step". The default is (False, 10). (10 cars)
    animation : boolean, optional
        If animation True, a gif gets created. BUT this option is only possibe if
        there are multiple plots (see multiplot arg in __data_provider), if multiplot
        in __data_provider is one either an error will occur or the gif shows a static
        picture (has not been tested!!!). The default is False.
    animation_save_dir : str, optional
        If animation is True then animation_save_dir will be used as the directory
        to save the produced gif, if nothing is proveded the gif gets saved in
        the same directory as this file. The default is False.
    interpolate :boolean, optional
        Provided animation is True, the KDE gets linearly interpolated between frames
        to get a smoother animation. The default is False.
    interpol_nframes : int, optional
        Provided animation is True as well as interpolate, the interpolate_frames
        gives the number of additional frames to be passed to the animation. The
        default is 3.

    Returns
    -------
    None.

    """
    # set title
    if title is None:
        if feature_number == "all":
            title = "Accidnets_in_Z??rich"
        else:
            title = features[feature_number]
    else:
        title = string_formater(title)
    title_add_lst = [(day_time[0], "day_time"), (seasons, "seasons"),
                      (temperature[0],"temperature"),(rain_dur[0],"rain_dur"),
                      (sumped[0],"sumped"), (sumbyc[0],"sumbyc"),(sumcars[0], "sumcars")]
    title_add = ""
    for i, item in enumerate(title_add_lst):
        if item[0]:
            title_add = title_add.join(str(item[1]))

    # set up init. data
    bunch_data_init, multiplots = __data_provider(df, features, feature_number,
                  feature_value, x_coord_number=x_coord_number, y_coord_number=y_coord_number,
                  severity_number=severity_number, title=title, date_number=date_number,
                  whole_data=whole_data, day_time=day_time, seasons=seasons,
                  temperature=temperature, rain_dur=rain_dur,  sumped=sumped, sumbyc=sumbyc, sumcars=sumcars)

    # init grid data
    X, Y = np.mgrid[BBox[0]:BBox[1]:100j, BBox[2]:BBox[3]:100j]

    # set up kde data
    bunch_data_kde, z_min_lst, z_max_lst = [], [], []
    for i in range(multiplots):
        data_kde, z_min_max = __kde_estimator(i, bunch_data_init, X, Y, visualize_seaborn,
                                          visualize_scipy, visualize_sklearn)
        bunch_data_kde.append(data_kde)
        z_min_lst.append(z_min_max[0])
        z_max_lst.append(z_min_max[1])

    # provide the levels for the contour plot, wont be needed for seaborn
    levels = np.linspace(0, max(z_max_lst), 100)[20:] # the slicing is done to turn of the last layers for visualiznig purposes #transparency

    # initialize custom colormap with transparency
    ccolormap(name='Blues_custom')

    # animated kde estimation
    if animation:
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='MarceloDave'), bitrate=2700)

        #set up KDE
        fig, ax = plt.subplots(figsize=(9, 10), dpi=144)
        if interpolate:
            interpol_steps = np.linspace(0, 1, interpol_nframes + 2)[1:][:-1]
            bunch_data_kde_new = []
            for i in range(multiplots-1):
                data, longitude, latitude, title_k, Z = bunch_data_kde[i]
                _, _, _, _, Z2 = bunch_data_kde[i + 1]
                for step in interpol_steps:
                    Z_interpol = __interpol(Z, Z2, step)
                    bunch_data_kde_new.append((data, longitude, latitude, title_k, Z_interpol))
            bunch_data_kde_new.append(bunch_data_kde[-1])
            bunch_data_kde = bunch_data_kde_new
            multiplots = (multiplots - 1) * interpol_nframes + 1

        #set up animation
        anim = matplotlib.animation.FuncAnimation(fig=fig, func=__visualizer,
                         frames=multiplots, fargs=(ax, bunch_data_kde,
                         X, Y, levels, im_map, BBox, features, severity_number,
                         title, visualize_real_data, visualize_seaborn, visualize_scipy,
                         visualize_sklearn, animation))


        #save file
        if animation_save_dir:
            anim.save(f"{animation_save_dir}\\{title}_{title_add}_{interpol_nframes}.mp4", writer=writer)
        else:
            anim.save(f"{title}_{interpol_nframes}.mp4", writer=writer)

    # regular kde estimation
    else:
        #set up KDE
        for i in range(multiplots):
            fig, ax = plt.subplots(dpi=120)
            __visualizer(i, ax, bunch_data_kde, X, Y, levels, im_map, BBox, features,
                         severity_number, title, visualize_real_data, visualize_seaborn, visualize_scipy,
                         visualize_sklearn, animation)


# this is a big function and not very pythonic, splitting would be recomended
# some functionality could be generalized for a more clear cut function i.e. all parameters beginning with "sum"
def __data_provider(df, features, feature_number, feature_value, x_coord_number=7,
                  y_coord_number=8, severity_number=2, title=None, date_number=0, whole_data=True, day_time=(False, 4), seasons=False,
                  temperature=(False, 5), rain_dur=(False, 15), sumped=(False, 10), sumbyc=(False, 10), sumcars=(False, 10)):

    """ Initialzes the data for kde estimation. See visualize_kde for description
    of the parameters."""

    if feature_number != "all":
        data = df[df[features[feature_number]] == feature_value]
    else:
        data = df

    cleand = lambda lst1, lst2: [item for i, item in enumerate(lst1) if lst2[i]]

    if whole_data:
        multiplots = 1
        titles = [title]
        x_coord = [data[features[x_coord_number]].values.reshape((-1, 1))]
        y_coord = [data[features[y_coord_number]].values.reshape((-1, 1))]
        data = [data]

    elif day_time[0]:
        multiplots = day_time[1]
        # hour_steps = 24/day_time[1]

        data[features[date_number]] = pd.to_datetime(data[features[date_number]])
        df_new = data[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]].copy()
        df_new.insert(3, 'Hour', df_new['Date'].dt.hour.to_numpy())

        hour_interval = list(np.arange(0, 24, int(24/day_time[1]))) + [24]
        df_new['DividedDay_category'] = pd.cut(df_new['Hour'], hour_interval, labels=range(day_time[1]))

        data = []
        x_coord = []
        y_coord = []
        titles = [f"{title}. Hours: {item}:00 - {(hour_interval + [0])[i+1]}:00" for i, item in enumerate(hour_interval)][:-1]
        for i in range(day_time[1]):
            unique_data_temp = df_new.groupby(['DividedDay_category']).get_group(i)
            data.append(unique_data_temp[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]])
            x_coord.append(unique_data_temp[features[x_coord_number]].values.reshape((-1, 1)))
            y_coord.append(unique_data_temp[features[y_coord_number]].values.reshape((-1, 1)))

    elif seasons:
        multiplots = 11
        data[features[date_number]] = pd.to_datetime(data[features[date_number]])
        df_new = data[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]].copy()
        df_new.insert(3, 'Month', df_new['Date'].dt.month.to_numpy())

        season_interval = np.linspace(1, 12, 12) # seasons shifted by one month
        df_new['DividedSeason_category'] = pd.cut(df_new['Month'], season_interval, labels=range(11))

        data = []
        x_coord = []
        y_coord = []
        seasons = ["Month 0-1", "Month 1-2", "Month 2-3", "Month 3-4", "Month 4-5",
                   "Month 5-6", "Month 6-7", "Month 7-8", "Month 8-9"
                   "Month 9-10", "Month 10-11", "Month 11-12"]
        titles = [f"{title}. {item} " for item in seasons]
        for i in range(11):
            unique_data_temp = df_new.groupby(['DividedSeason_category']).get_group(i)
            data.append(unique_data_temp[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]])
            x_coord.append(unique_data_temp[features[x_coord_number]].values.reshape((-1, 1)))
            y_coord.append(unique_data_temp[features[y_coord_number]].values.reshape((-1, 1)))

    elif temperature[0]:
        data[features[date_number]] = pd.to_datetime(data[features[date_number]])
        df_new = data[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number], features[9]]].copy()

        max_temp = df_new[features[9]].max()
        temp_interval = list(np.arange(df_new[features[9]].min(), max_temp, temperature[1])) + [max_temp]
        df_new['DividedTemp_category'] = pd.cut(df_new[features[9]], temp_interval, labels=range(len(temp_interval)-1))
        unique_category_temp = df_new.groupby(['DividedTemp_category']).count()
        unique_category = unique_category_temp.index.to_numpy()
        occurences = unique_category_temp[features[9]].to_numpy()
        unique_category = cleand(unique_category, occurences)

        multiplots = len(unique_category)
        data = []
        x_coord = []
        y_coord = []
        titles = []
        for index, i in enumerate(unique_category):
            unique_data_temp = df_new.groupby(['DividedTemp_category']).get_group(i)
            data.append(unique_data_temp[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]])
            x_coord.append(unique_data_temp[features[x_coord_number]].values.reshape((-1, 1)))
            y_coord.append(unique_data_temp[features[y_coord_number]].values.reshape((-1, 1)))

            pervious_index = unique_category[index-1] if index else -1
            if int(i - pervious_index) == 1:
                titles.append(f"{title}. Average temperature range: {round(temp_interval[i], 2)}-{round((temp_interval + [max_temp])[i+1], 2)} ??C")
            else:
                titles.append(f"{title}. Average temperature range: {round(temp_interval[unique_category[index-1]], 2)}-{round((temp_interval + [max_temp])[i+1], 2)} ??C")

    elif rain_dur[0]:
        data[features[date_number]] = pd.to_datetime(data[features[date_number]])
        df_new = data[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number], features[10]]].copy()

        max_dur = 60
        dur_interval = list(np.arange(0, max_dur, rain_dur[1])) + [max_dur]
        df_new['DividedRain_category'] = pd.cut(df_new[features[10]], dur_interval, labels=range(len(dur_interval)-1))
        unique_category_temp = df_new.groupby(['DividedRain_category']).count()
        unique_category = unique_category_temp.index.to_numpy()
        occurences = unique_category_temp[features[10]].to_numpy()
        unique_category = cleand(unique_category, occurences)

        multiplots = len(unique_category)
        data = []
        x_coord = []
        y_coord = []
        titles = []
        for index, i in enumerate(unique_category):
            unique_data_temp = df_new.groupby(['DividedRain_category']).get_group(i)
            data.append(unique_data_temp[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]])
            x_coord.append(unique_data_temp[features[x_coord_number]].values.reshape((-1, 1)))
            y_coord.append(unique_data_temp[features[y_coord_number]].values.reshape((-1, 1)))

            pervious_index = unique_category[index-1] if index else -1
            if int(i - pervious_index) == 1:
                titles.append(f"{title}. Average rain duration range: {round(dur_interval[i], 2)}-{round((dur_interval + [max_dur])[i+1], 2)} min")
            else:
                titles.append(f"{title}. Average rain duration range: {round(dur_interval[unique_category[index-1]], 2)}-{round((dur_interval + [max_dur])[i+1], 2)} min")


    elif sumped[0]:
        data[features[date_number]] = pd.to_datetime(data[features[date_number]])
        df_new = data[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number], features[12]]].copy()
        df_new.dropna(inplace=True)

        max_ped = df_new[features[12]].max()
        ped_interval = list(np.arange(df_new[features[12]].min(), max_ped, sumped[1])) + [max_ped]
        df_new['DividedPed_category'] = pd.cut(df_new[features[12]], ped_interval, labels=range(len(ped_interval)-1))
        unique_category_temp = df_new.groupby(['DividedPed_category']).count()
        unique_category = unique_category_temp.index.to_numpy()
        occurences = unique_category_temp[features[12]].to_numpy()
        unique_category = cleand(unique_category, occurences)

        multiplots = len(unique_category)
        data = []
        x_coord = []
        y_coord = []
        titles = []
        for index, i in enumerate(unique_category):
            unique_data_temp = df_new.groupby(['DividedPed_category']).get_group(i)
            data.append(unique_data_temp[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]])
            x_coord.append(unique_data_temp[features[x_coord_number]].values.reshape((-1, 1)))
            y_coord.append(unique_data_temp[features[y_coord_number]].values.reshape((-1, 1)))

            pervious_index = unique_category[index-1] if index else -1
            if int(i - pervious_index) == 1:
                titles.append(f"{title}. Number of Pedestrians range: {round(ped_interval[i])}-{round((ped_interval + [max_ped])[i+1])}")
            else:
                titles.append(f"{title}. Number of Pedestrians range: {round(ped_interval[unique_category[index-1]])}-{round((ped_interval + [max_ped])[i+1])}")

    elif sumbyc[0]:
        data[features[date_number]] = pd.to_datetime(data[features[date_number]])
        df_new = data[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number], features[11]]].copy()
        df_new.dropna(inplace=True)

        max_byc = df_new[features[11]].max()
        byc_interval = list(np.arange(df_new[features[11]].min(), max_byc, sumbyc[1])) + [max_byc]
        df_new['DividedByc_category'] = pd.cut(df_new[features[11]], byc_interval, labels=range(len(byc_interval)-1))
        unique_category_temp = df_new.groupby(['DividedByc_category']).count()
        unique_category = unique_category_temp.index.to_numpy()
        occurences = unique_category_temp[features[11]].to_numpy()
        unique_category = cleand(unique_category, occurences)

        multiplots = len(unique_category)
        data = []
        x_coord = []
        y_coord = []
        titles = []
        for index, i in enumerate(unique_category):
            unique_data_temp = df_new.groupby(['DividedByc_category']).get_group(i)
            data.append(unique_data_temp[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]])
            x_coord.append(unique_data_temp[features[x_coord_number]].values.reshape((-1, 1)))
            y_coord.append(unique_data_temp[features[y_coord_number]].values.reshape((-1, 1)))

            pervious_index = unique_category[index-1] if index else -1
            if int(i - pervious_index) == 1:
                titles.append(f"{title}. Number of bicycles range: {round(byc_interval[i])}-{round((byc_interval + [max_byc])[i+1])}")
            else:
                titles.append(f"{title}. Number of bicycles range: {round(byc_interval[unique_category[index-1]])}-{round((byc_interval + [max_byc])[i+1])}")

    elif sumcars[0]:
        data[features[date_number]] = pd.to_datetime(data[features[date_number]])
        df_new = data[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number], features[13]]].copy()
        df_new.dropna(inplace=True)

        max_cars = df_new[features[13]].max()
        cars_interval = list(np.arange(df_new[features[13]].min(), max_cars, sumcars[1])) + [max_cars]
        df_new['DividedCar_category'] = pd.cut(df_new[features[13]], cars_interval, labels=range(len(cars_interval)-1))
        unique_category_temp = df_new.groupby(['DividedCar_category']).count()
        unique_category = unique_category_temp.index.to_numpy()
        occurences = unique_category_temp[features[13]].to_numpy()
        unique_category = cleand(unique_category, occurences)

        multiplots = len(unique_category)
        data = []
        x_coord = []
        y_coord = []
        titles = []
        for index, i in enumerate(unique_category):
            unique_data_temp = df_new.groupby(['DividedCar_category']).get_group(i)
            data.append(unique_data_temp[[features[date_number], features[severity_number], features[x_coord_number], features[y_coord_number]]])
            x_coord.append(unique_data_temp[features[x_coord_number]].values.reshape((-1, 1)))
            y_coord.append(unique_data_temp[features[y_coord_number]].values.reshape((-1, 1)))

            pervious_index = unique_category[index-1] if index else -1
            if int(i - pervious_index) == 1:
                titles.append(f"{title}. Number of cars range: {round(cars_interval[i])}-{round((cars_interval + [max_cars])[i+1])}")
            else:
                titles.append(f"{title}. Number of cars range: {round(cars_interval[unique_category[index-1]])}-{round((cars_interval + [max_cars])[i+1])}")


    return (data, x_coord, y_coord, titles), multiplots


def __kde_estimator(i, bunch_data_init, X, Y, visualize_seaborn, visualize_scipy,
                    visualize_sklearn):

    """ Helper function. Calculate KDE. See visualize_kde for description
    of the parameters."""
    # set up data
    data, x_coord, y_coord, title = bunch_data_init[0][i], bunch_data_init[1][i], bunch_data_init[2][i], bunch_data_init[3][i]
    longitude, latitude = lv95_latlong(x_coord, y_coord)

    if visualize_seaborn:
        # set dummy data
        Z = 0
        z_min_max = (0, 0)

    if visualize_scipy:
        kernel = stats.gaussian_kde([longitude[:, 0], latitude[:,0]], bw_method="silverman")
        Z = np.reshape(kernel([X.ravel(), Y.ravel()]).T, X.shape)
        z_min_max = (Z.min(), Z.max())

    if visualize_sklearn:
        # formating data
        samples = np.vstack([longitude[:, 0], latitude[:,0]]).T
        xy_grid = np.vstack([X.ravel(), Y.ravel()]).T
        bandwidths = np.linspace(0.0, 0.1, 256)

        #setting up a grid search to find best bandwidth via 2 fold corss validation
        grid = GridSearchCV(KernelDensity(kernel='gaussian', algorithm="auto"), {'bandwidth': bandwidths},
                            cv=KFold(2))

        grid = grid.fit(samples) # perform estimation
        kde =  grid.best_estimator_ # find estimation for best suited bandwidth
        # print(f"Best bandwidth: {kde.bandwidth}")

        # set up grid data
        Z = kde.score_samples(xy_grid) # score_samples returns the log-likelihood of the sample
        Z = np.exp(Z.reshape(X.shape))
        z_min_max = (Z.min(), Z.max())
        # print(stats.kstest(xy_grid, kde))

    return (data, longitude, latitude,title, Z), z_min_max


# this has been writen as a function to do easy animations
def __visualizer(i, ax, bunch_data_kde, X, Y, levels, im_map, BBox,
               features, severity_number, title, visualize_real_data,
               visualize_seaborn, visualize_scipy, visualize_sklearn,
               animation):
    """Helper function. Plot data. See visualize_kde for description
    of the parameters."""

    data, longitude, latitude, title, Z = bunch_data_kde[i]
    ax.clear()
    if severity_number is not False:
        severity = data[features[severity_number]].values
        sizes = list((0.1*severity))
    else:
        sizes = list(np.ones(len(longitude)))

    if visualize_seaborn:
        sns.set_style("whitegrid")
        h = sns.jointplot(longitude[:,0], latitude[:,0], kind="kde", fill=True)
        # h = sns.jointplot(longitude[:,0], latitude[:,0], color="k", marker='.').plot_joint(sns.kdeplot, n_levels=20, shade=True, alpha=0.6)
        h.set_axis_labels('Longitude [??]', 'Latitude [??]', fontsize=16)
        plt.suptitle(title)


    if visualize_scipy:
        ax.imshow(im_map, extent=BBox, aspect='auto')
        co = ax.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Blues, alpha=0.5)
        if visualize_real_data:
            ax.scatter(longitude, latitude, color='k', s=sizes, alpha=0.5)
        # cbar = plt.colorbar(co/500)
        plt.title(title)
        plt.xlabel("Longitude [??]")
        plt.ylabel("Latitude [??]")
        if not animation:
            plt.show()


    if visualize_sklearn:
        ax.imshow(im_map, extent=BBox, aspect='auto')
        ax.contourf(X, Y, Z, levels=levels, cmap="hot_r",  alpha=0.5, antialiased=True)
        print(f'Image#: {i}')
        if visualize_real_data:
            ax.scatter(longitude, latitude, s=sizes)
        plt.title(title)
        # cbar = plt.colorbar(co)
        plt.xlabel("Longitude [??]")
        plt.ylabel("Latitude [??]")
        if not animation:
            plt.show()


def string_formater(string):
    """Formating string for file title."""
    special_char = [' ', ':', '-', '<', '>', '.', ',', ';', '/']
    for char in special_char:
        string = string.replace(char, '_')
    return string


def __interpol(m1, m2, delta_t):
    """Linear interpolation between matrices.

    Parameter
    ---------
    m1 : array like (NxN)
        Matrix 1. For us this is the Z coordinates as given from the kde estimation.
    m1 : array like (NxN)
        Matrix 2. For us this is the Z coordinates as given from the kde estimation.
    delta_t : float element of ]0, 1[
        Float a value between 0 and 1.

    Reference
    ---------
    .. [1] sebastian, stackoverflow, (Sep 6 2013 at 9:51), [online]
       "**Interpolation between two matrices**". Aviable at:
       https://stackoverflow.com/questions/18654511/interpolation-between-two-matrices ,
       [Accessed 18 Mai 2021]
    """
    return m1 + (m2 - m1) * delta_t

# =============================================================================
# Display KDE
# =============================================================================
# the following part could have been written much more elegantly, but lets do it with brute force
if __name__ == "__main__":

    titles_invo = ["KDE: accidents involving pedestrians, 2011-2020",
                  "KDE: accidents involving bicycles, 2011-2020",
                  "KDE: accidents involving motocycles, 2011-2020",]


    lst_on_roadt = [0, 1, 2, 3, 4, 9]
    titles_on_roadt = ["KDE: accidents on motorways, 2011-2020",
                  "KDE: accidents on expressways, 2011-2020",
                  "KDE: accidents on principal roads, 2011-2020",
                  "KDE: accidents on minor roads, 2011-2020",
                  "KDE: accidents on motorways side installation, 2011-2020",
                  "KDE: accidents on other road types, 2011-2020"]

    l_acct = [1, 2, 3, 4, 5, 6, 7, 8, 9] # check ambiguity of 0 and 00 in accident type (pdf)
    titles_acct = ["KDE: accidents when overtaking or changing lanes, 2011-2020",
                  "KDE: accidents with rear-end collision, 2011-2020",
                  "KDE: accidents when turning left or right, 2011-2020",
                  "KDE: accidents when turning-into main road, 2011-2020",
                  "KDE: accidents when crossing the lane(s), 2011-2020",
                  "KDE: accidents with head-on collision, 2011-2020",
                  "KDE: accidents when parking, 2011-2020",
                  "KDE: accidents involving pedestrian(s), 2011-2020",
                  "KDE: accidents involving animal(s), 2011-2020"]

    whole_data = False
    day_time = (False, 4)
    seasons = True
    temperature = (False, 3)
    rain_dur = (False, 15)
    sumped = (False, 200)
    sumbyc = (False, 69)
    sumcars = (False, 100)

    animate_daytime = False
    interpol_nframes = 15
    animation_save_dir="Figures"

# =============================================================================
    if whole_data:
        #involvements
        visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0])
        # visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1])
        # visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2])

        # # road type
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0])
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1])
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2])
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3])
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4])
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5])

        # # accident type
        # visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0])
        # visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1])
        # visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2])
        # visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3])
        # # visualize_kde(df, map00, BBox, features, 2, l_acct[4], title=titles_acct[4]) # not enough data points for this estimation
        # # visualize_kde(df, map00, BBox, features, 2, l_acct[5], title=titles_acct[5]) # not enough data points for this estimation
        # # visualize_kde(df, map00, BBox, features, 2, l_acct[6], title=titles_acct[6]) # not enough data points for this estimation
        # # visualize_kde(df, map00, BBox, features, 2, l_acct[7], title=titles_acct[7]) # not enough data points for this estimation
        # # visualize_kde(df, map00, BBox, features, 2, l_acct[8], title=titles_acct[8]) # not enough data points for this estimation

# =============================================================================
    if day_time[0]:
        # involvements
        visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False, day_time=day_time)

        # road type
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False, day_time=day_time)

        # accident type
        visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False, day_time=day_time)
        visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False, day_time=day_time)
        # visualize_kde(df, map00, BBox, features, 2, l_acct[4], title=titles_acct[4]) # not enough data points for this estimation
        # visualize_kde(df, map00, BBox, features, 2, l_acct[5], title=titles_acct[5]) # not enough data points for this estimation
        # visualize_kde(df, map00, BBox, features, 2, l_acct[6], title=titles_acct[6]) # not enough data points for this estimation
        # visualize_kde(df, map00, BBox, features, 2, l_acct[7], title=titles_acct[7]) # not enough data points for this estimation
        # visualize_kde(df, map00, BBox, features, 2, l_acct[8], title=titles_acct[8]) # not enough data points for this estimation
# =============================================================================

    if animate_daytime:
        #involvements
        visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False,
                        day_time=day_time, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False,
                        day_time=day_time, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # road type
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=4, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        #accident type
        visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False,
                      day_time=day_time, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
# =============================================================================

    if seasons:
        # visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False,
                        seasons=True, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False,
        #                 seasons=True, animation=True, visualize_real_data=False,
        #                 interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # # road type
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False,
        # #               seasons=True, animation=True, visualize_real_data=False,
        # #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # #accident type
        # visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False,
        #               seasons=True, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
# =============================================================================

    if temperature[0]:
        visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False,
                        temperature=temperature, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False,
        #                 temperature=temperature, animation=True, visualize_real_data=False,
        #                 interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        #road type
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=4, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False,
        #               temperature=temperature, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False,
        #               temperature=temperature, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # #accident type
        # visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False,
        #               temperature=temperature, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False,
                      temperature=temperature, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
# =============================================================================

    if rain_dur[0]:
        visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False,
                        rain_dur=rain_dur, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False,
                        rain_dur=rain_dur, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # road type
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=4, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False,
        #               rain_dur=rain_dur, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False,
        #               rain_dur=rain_dur, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        #accident type
        # visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False,
        #               rain_dur=rain_dur, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False,
                      rain_dur=rain_dur, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
# =============================================================================

    if sumped[0]:
        visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False,
                      sumped=sumped, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False,
                        sumped=sumped, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False,
                        sumped=sumped, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # road type
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False,
        #               sumped=sumped, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=4, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False,
        #               sumped=sumped, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False,
        #               sumped=sumped, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False,
                      sumped=sumped, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False,
        #               sumped=sumped, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False,
        #               sumped=sumped, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # accident type
        visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False,
                      sumped=sumped, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False,
                      sumped=sumped, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False,
                      sumped=sumped, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False,
                      sumped=sumped, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
# =============================================================================

    if sumbyc[0]:
        # visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False,
        #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False,
        #                 sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #                 interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False,
        #                 sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #                 interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # # road type
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False,
        #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=4, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False,
        #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False,
        #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False,
        # #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        # #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False,
        #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # # visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False,
        # #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        # #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        #accident type
        # visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False,
        #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        # visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False,
        #               sumbyc=sumbyc, animation=True, visualize_real_data=False,
        #               interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False,
                      sumbyc=sumbyc, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False,
                      sumbyc=sumbyc, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
# =============================================================================

    if sumcars[0]:
        visualize_kde(df, map00, BBox, features, 3, 1, title=titles_invo[0], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 4, 1, title=titles_invo[1], whole_data=False,
                        sumcars=sumcars, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 5, 1, title=titles_invo[2], whole_data=False,
                        sumcars=sumcars, animation=True, visualize_real_data=False,
                        interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        # road type
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[0], title=titles_on_roadt[0], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=4, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[1], title=titles_on_roadt[1], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[2], title=titles_on_roadt[2], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[3], title=titles_on_roadt[3], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[4], title=titles_on_roadt[4], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 6, lst_on_roadt[5], title=titles_on_roadt[5], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)

        #accident type
        visualize_kde(df, map00, BBox, features, 2, l_acct[0], title=titles_acct[0], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[1], title=titles_acct[1], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[2], title=titles_acct[2], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
        visualize_kde(df, map00, BBox, features, 2, l_acct[3], title=titles_acct[3], whole_data=False,
                      sumcars=sumcars, animation=True, visualize_real_data=False,
                      interpolate=True, interpol_nframes=interpol_nframes, animation_save_dir=animation_save_dir)
# =============================================================================
