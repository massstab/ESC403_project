# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:30:42 2021
@author: marszpd, dtm
"""
import numpy as np
import pandas as pd
import time

# =============================================================================
# Tidy data functions
# =============================================================================
def to_int(data, feature_list, m, k):
    """
    Removes unwanted string part of the data.
    :param data: Pandas data frame.
    :param feature_list: List of features.
    :param m: Feature number, as saved in the list feature_list.
    :param k: Length of the unnecessary string.
    :return: None
    """

    data_arr = data[feature_list[m]].to_numpy()
    data[feature_list[m]] = [int(item[k:]) for item in data_arr]  # make str of form "str" + "number" to number


def find_day(df):
    """
    Replaces the year, month, weekday columns in accident df to a pandas datetime object as index.
    :param df: the accident data frame
    :return: the new data frame with added datetime column
    """
    new_df = df.copy()
    new_df['Date'] = np.nan
    new_df.drop(['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=1, inplace=True)
    month = 1
    weekday = df['AccidentWeekDay'].values[0]
    day = 1
    idx = 0
    for row in df[['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour']].values:
        year, hour = row[0], row[3]
        if month != row[1]:
            print(year, date.month_name())
            month = row[1]
            day = 0
        if weekday != row[2]:
            weekday = row[2]
            day += 1
        try:
            date = pd.to_datetime(str(year) + "-" + str(month) + "-" + str(day) + "-" + str(hour) + ':00')
        except ValueError:
            h_last = new_df.iloc[idx - 1]['Date'].hour  # Set the hour of the last accident if value is nan
            date = pd.to_datetime(str(year) + "-" + str(month) + "-" + str(day) + "-" + str(h_last) + ':00')
        new_df.at[idx, 'Date'] = date
        idx += 1
    new_df.set_index('Date', inplace=True)

    return new_df


def meteo_date_prep(df):
    """
    Brings the meteo raw_data data in the right format and adds the datetime as index to the new
    df and computes the average temperature/rain duration from the measuring stations
    :param df: raw_data meteo dataframe
    :return: the new meteo dataframe
    """
    new_df = pd.DataFrame(columns=['Date', 'AvgTemperature', 'AvgRainDur'], dtype=float)
    i = 0
    temp_T = 0.0
    temp_Rain = 0.0
    visited = False
    num_of_stations = 0
    current_hour = '00'
    first_station = df['Standort'].values[0]
    start_time = df['Datum'].values[0][11:13]
    if start_time != current_hour:
        print('Please start the data frame at time 00:00')
    for row in df[['Datum', 'Standort', 'Parameter', 'Wert']].values:
        if row[0][11:13] != current_hour:
            new_df.at[i, 'AvgTemperature'] = round(temp_T / num_of_stations, 1)
            new_df.at[i, 'AvgRainDur'] = round(temp_Rain / num_of_stations, 1)
            i += 1  # Update index for the next entry in new_df
            temp_T = 0.0
            temp_Rain = 0.0
            visited = False
            num_of_stations = 0
            current_hour = row[0][11:13]
            first_station = row[1]
            current_station = first_station
        if (row[2] == 'T') or (row[2] == 'RainDur'):
            if (row[2] == 'RainDur'):
                temp_Rain += row[3]
                continue
            if not visited:
                visited = True
                num_of_stations += 1
                temp_T += row[3]
                year, month, day, hour = row[0][:4], row[0][5:7], row[0][8:10], row[0][11:13]
                date = pd.to_datetime(year + "-" + month + "-" + day + "-" + hour)
                new_df.at[i, 'Date'] = date
                continue

            current_station = row[1]
            if current_station != first_station:
                if (row[2] == 'RainDur'):
                    temp_Rain += row[3]
                    continue
                num_of_stations += 1
                temp_T += row[3]
                continue

    new_df.at[i, 'AvgTemperature'] = round(temp_T / num_of_stations, 1)  # The last entry
    new_df.at[i, 'AvgRainDur'] = round(temp_Rain / num_of_stations, 1)  # The last entry
    new_df.set_index('Date', inplace=True)
    return new_df


def velo_fuss_date_prep(df):
    """
    Brings the bike and pedestrian raw data in the right format and computes
    the sum of bikes and pedestrains (in both street directions) passing
    the corresponding detector.

    Parameters
    ----------
    df : Pandas dataframe
        Raw bike and pedetsrian count dataframe.

    Returns
    -------
    New bike and pedestrain dataframe, where the per quarter hour count has been summed
    to give per hour counts.
    """
    new_df = pd.DataFrame(columns=['Date','AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N',
               'SumBikerNumber', 'SumPedastrianNumber'])

    k = 0
    id_lst = list(set(df['FK_STANDORT'].to_numpy())) # to be pedantic, it's not the id but the location id
    for i, id_number in enumerate(id_lst):
        data_i = df[df['FK_STANDORT'] == id_number].values # gives a dataframe with just the data from that specific id_number
        number_data_points = data_i.shape[0]
        year = data_i[0][2][:4]

        j = 0
        loop_count = 0
        while j <= number_data_points - 1:

            # set date as done by the function meteo_date_prep, well at least as done by the old version
            month, day, hour = data_i[j][2][5:7], data_i[j][2][8:10], data_i[j][2][11:13]
            date = pd.to_datetime(year + "-" + month + "-" + day + "-" + hour)
            new_df.at[k + loop_count, 'Date'] = date

            # set coordinates
            new_df.at[k + loop_count, 'AccidentLocation_CHLV95_E'] = data_i[0][7]
            new_df.at[k + loop_count, 'AccidentLocation_CHLV95_N'] = data_i[0][8]

            # search and resolve count inconsistancies, i.e. if there are missing measurements
            hours = [item[11:13] for item in list(data_i[j:j+4][:,2])]
            occurrence = hours.count(hours[0])

            # set sum of the data
            new_df.at[k + loop_count, 'SumBikerNumber'] = __helper_velo_fuss(data_i[j:j+occurrence][:,3], data_i[j:j+occurrence][:,4])
            new_df.at[k + loop_count, 'SumPedastrianNumber'] = __helper_velo_fuss(data_i[j:j+occurrence][:,5], data_i[j:j+occurrence][:,6])

            loop_count += 1
            j += occurrence
        k += loop_count
    new_df.set_index('Date', inplace=True)
    return new_df


def __helper_velo_fuss(lst1, lst2):
    """ Helper function for velo_fuss_date_prep."""
    nan_bool_lst1 = list(pd.Series(lst1).isnull()) # if an element is True, the corresponding element in lst is nan
    nan_bool_lst2 = list(pd.Series(lst2).isnull()) # if an element is True, the corresponding element in lst is nan

    reduce = lambda lst, nan_lst: [item for i, item in enumerate(lst) if not nan_lst[i]]
    red1 = reduce(lst1, nan_bool_lst1)
    red2 = reduce(lst2, nan_bool_lst2)

    if red1 == [] and red2 == []:
        return np.nan
    else:
        # for n missing points, interpolate by taking the last value n times,
        # if the length of the lists are smaller than 4. One can off course use
        # linear regression (if we have a minimum of two points), but I think
        # this would be overkill since the data has a relatively high measurement error,
        # and using only 3 or less data points for regression isn't worth it I think.
        if red1 == [] and 0 < len(red2): # it turns out that the inequality i.e. <= isn't neccessary since one can see that once one value is nan all 3 others are too
            return sum(red2) + red2[-1]*len(red2)*(len(red2) < 4)
        elif red2 == [] and 0 < len(red1): # it turns out that the inequality i.e. <= isn't neccessary since one can see that once one value is nan all 3 others are too
            return sum(red1) + red1[-1]*len(red1)*(len(red1) < 4)
        else:
            return 0.5 * (sum(red1 + red2) + red1[-1]*len(red1)*(len(red1) < 4) + red2[-1]*len(red2)*(len(red2) < 4))


def auto_prep_to_pickel(df, file_name, directory):
    df.to_pickle(f"{directory}/{file_name}.pickle")


# this probably takes quiet a long time to be executed
def associate_coord_to_accident_coord(radius, df1, df2):
    """
    Alocates the nearest location with respect to the locations in df1, if the
    distances of two or more points relative to the reference point are equal
    the data will be averaged. It is assumed that the coordinates are in close
    proximaty s.t. the coordinate systems is locally flat.

    Parameters
    ----------
    radius : float
        l2 maximal distance (in meters) from one reference point ind df1 to a
        point in df2. If the distance between the point in df1 and df2 is greater
        than the radius, the point in df2 will not be associated to the reference point.
    df1 : DataFrame
        Main dataframe (accident data), the data from df2 will be inserted into
        this one.
    df2 : DataFrame
        Data for merging with df1.

    Returns
    -------
    Dataframe which matches both the coordinates in df1 and df2 up to a distance smaller
    or equal to radius, and the date, i.e. the rows to change will be appended to the
    new dataframe.
    """

    # find the unique coordinates
    unique_coord_df1_temp = df1.groupby(['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N'])
    unique_coord_df1 = unique_coord_df1_temp.count().index

    unique_coord_df2_temp = df2.groupby(['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N'])
    unique_coord_df2 = unique_coord_df2_temp.count().index
    unique_coord_df2_arr = np.array([list(item) for item in unique_coord_df2]) # the items are toopls, to_numpy() does not convert it properly to the wanted format

    new_df = pd.DataFrame(columns=['Date','AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N',
                                   'SumBikerNumber', 'SumPedestrianNumber'])

    for coord1 in unique_coord_df1:
        coord2_lst = []
        diff_temp = np.subtract(coord1, unique_coord_df2_arr)
        distances = np.sqrt(np.sum(diff_temp*diff_temp,  axis=-1))
        indices_minima = np.where(distances == min(distances))[0]
        if distances[indices_minima[0]] <= radius:
            coord2_lst.extend(list(unique_coord_df2[indices_minima].to_numpy()))
            if len(indices_minima) > 1:
                print("mo")

        # if there are coordinates within the radius, find the corresponding dates if aviable and overwrite the previously given nan
        # if there are more than one coord. in coord2_lst take the average
        if len(coord2_lst) != 0:
            indices1 = unique_coord_df1_temp.get_group(coord1).index

            for i in indices1:
                aviable_date = df1['Date'].iloc[i]
                val_ped, val_bike = [], []
                for item in coord2_lst:
                    if aviable_date in unique_coord_df2_temp.get_group(item)['Date'].values:
                        data_j = unique_coord_df2_temp.get_group(item)[unique_coord_df2_temp.get_group(item)['Date'] == aviable_date].values
                        val_bike.append(data_j[0][3])
                        val_ped.append(data_j[0][4])

                # format to drop nan
                val_bike = np.array(val_bike)
                val_ped = np.array(val_ped)

                val_bike = val_bike[np.logical_not(np.isnan(val_bike))]
                val_ped = val_ped[np.logical_not(np.isnan(val_ped))]

                length_bike = val_bike.shape[0]
                length_ped = val_ped.shape[0]

                if length_bike != 0 or length_ped != 0:

                    val_bike = val_bike.sum()/length_bike if length_bike else np.nan # take the average if there are multiple points with the same distance
                    val_ped = val_ped.sum()/length_ped if length_ped else np.nan # take the average if there are multiple points with the same distance

                    date = pd.to_datetime(aviable_date)
                    new_df.at[i, 'Date'] = date

                    new_df.at[i, 'AccidentLocation_CHLV95_E'] = coord1[0]
                    new_df.at[i, 'AccidentLocation_CHLV95_N'] = coord1[1]

                    new_df.at[i, 'SumBikerNumber'] = val_bike
                    new_df.at[i, 'SumPedestrianNumber'] = val_ped

    new_df.set_index('Date', inplace=True)
    return new_df


# sorry lazy copy of the function before for car count data
def associate_coord_to_accident_coord_cars(radius, df1, df2):
    """
    Alocates the nearest location with respect to the locations in df1, if the
    distances of two or more points relative to the reference point are equal
    the data will be averaged. It is assumed that the coordinates are in close
    proximaty s.t. the coordinate systems is locally flat.

    Parameters
    ----------
    radius : float
        l2 maximal distance (in meters) from one reference point ind df1 to a
        point in df2. If the distance between the point in df1 and df2 is greater
        than the radius, the point in df2 will not be associated to the reference point.
    df1 : DataFrame
        Main dataframe (accident data), the data from df2 will be inserted into
        this one.
    df2 : DataFrame
        Data for merging with df1.

    Returns
    -------
    Dataframe which matches both the coordinates in df1 and df2 up to a distance smaller
    or equal to radius, and the date, i.e. the rows to change will be appended to the
    new dataframe.
    """

    # find the unique coordinates
    unique_coord_df1_temp = df1.groupby(['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N'])
    unique_coord_df1 = unique_coord_df1_temp.count().index

    unique_coord_df2_temp = df2.groupby(['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N'])
    unique_coord_df2 = unique_coord_df2_temp.count().index
    unique_coord_df2_arr = np.array([list(item) for item in unique_coord_df2]) # the items are toopls, to_numpy() does not convert it properly to the wanted format

    new_df = pd.DataFrame(columns=['Date','AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N',
                                   'SumCars'])

    for coord1 in unique_coord_df1:
        coord2_lst = []
        diff_temp = np.subtract(coord1, unique_coord_df2_arr)
        distances = np.sqrt(np.sum(diff_temp*diff_temp,  axis=-1))
        indices_minima = np.where(distances == min(distances))[0]
        if distances[indices_minima[0]] <= radius:
            coord2_lst.extend(list(unique_coord_df2[indices_minima].to_numpy()))

        # if there are coordinates within the radius, find the corresponding dates if aviable and overwrite the previously given nan
        # if there are more than one coord. in coord2_lst take the average
        if len(coord2_lst) != 0:
            indices1 = unique_coord_df1_temp.get_group(coord1).index

            for i in indices1:
                aviable_date = df1['Date'].iloc[i]
                val_car = []
                for item in coord2_lst:
                    if aviable_date in unique_coord_df2_temp.get_group(item)['Date'].values:
                        data_j = unique_coord_df2_temp.get_group(item)[unique_coord_df2_temp.get_group(item)['Date'] == aviable_date].values
                        val_car.append(data_j[0][3])

                # format to drop nan
                val_car = np.array(val_car)
                val_car = val_car[np.logical_not(np.isnan(val_car))]
                length_car = val_car.shape[0]

                if length_car != 0:

                    val_car = val_car.sum()/length_car if length_car else np.nan # take the average if there are multiple points with the same distance

                    date = pd.to_datetime(aviable_date)
                    new_df.at[i, 'Date'] = date

                    new_df.at[i, 'AccidentLocation_CHLV95_E'] = coord1[0]
                    new_df.at[i, 'AccidentLocation_CHLV95_N'] = coord1[1]

                    new_df.at[i, 'SumCars'] = val_car

    new_df.set_index('Date', inplace=True)
    return new_df


# =============================================================================
# Raw data
# =============================================================================
# data_accident = pd.read_csv("raw_data/RoadTrafficAccidentLocations.csv")

features_original_accident = ['AccidentUID', 'AccidentType', 'AccidentType_de',
                              'AccidentType_fr', 'AccidentType_it', 'AccidentType_en',
                              'AccidentSeverityCategory', 'AccidentSeverityCategory_de',
                              'AccidentSeverityCategory_fr', 'AccidentSeverityCategory_it',
                              'AccidentSeverityCategory_en', 'AccidentInvolvingPedestrian',
                              'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle',
                              'RoadType', 'RoadType_de', 'RoadType_fr', 'RoadType_it',
                              'RoadType_en', 'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N',
                              'CantonCode', 'MunicipalityCode', 'AccidentYear', 'AccidentMonth',
                              'AccidentMonth_de', 'AccidentMonth_fr', 'AccidentMonth_it',
                              'AccidentMonth_en', 'AccidentWeekDay', 'AccidentWeekDay_de',
                              'AccidentWeekDay_fr', 'AccidentWeekDay_it', 'AccidentWeekDay_en',
                              'AccidentHour', 'AccidentHour_text']

features_croped_accident = ['AccidentYear', 'AccidentMonth', 'AccidentWeekDay',
                            'AccidentHour', 'AccidentType',
                            'AccidentSeverityCategory', 'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle',
                            'AccidentInvolvingMotorcycle', 'RoadType', 'AccidentLocation_CHLV95_E',
                            'AccidentLocation_CHLV95_N']
# =============================================================================
# data_meteo = pd.read_csv("raw_data/ugz_ogd_meteo_h1_2011-2020.csv")

features_meteo = ['Datum', 'Standort', 'Parameter', 'Intervall', 'Einheit',
                  'Wert', 'Status']

# =============================================================================
# caution relatively big files
# data_velo_fussgang11 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2011_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang12 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2012_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang13 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2013_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang14 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2014_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang15 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2015_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang16 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2016_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang17 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2017_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang18 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2018_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang19 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2019_verkehrszaehlungen_werte_fussgaenger_velo.csv")
# data_velo_fussgang20 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2020_verkehrszaehlungen_werte_fussgaenger_velo.csv")

features_velo_fuss = ['FK_ZAEHLER', 'FK_STANDORT', 'DATUM', 'VELO_IN', 'VELO_OUT',
                      'FUSS_IN','FUSS_OUT', 'OST', 'NORD']

features_croped_velo_fuss = ['FK_STANDORT', 'DATUM', 'VELO_IN', 'VELO_OUT',
                      'FUSS_IN','FUSS_OUT', 'OST', 'NORD']

features_new_velo_fuss = ['AccidentYear', 'AccidentMonth', 'AccidentWeekDay',
                          'AccidentHour', 'AccidentLocation_CHLV95_E',
                          'AccidentLocation_CHLV95_N' 'SumBikerNumber',
                          'SumPedastrianNumber']

# =============================================================================
# Perform cleaning
# =============================================================================
# clean accident data
# to generate all accident tidy_data data, uncomment this section
"""
data_accident_cleaned = data_accident[features_croped_accident].copy()  # complete new df with only used features
data_accident_cleaned[features_original_accident[11:14]] = data_accident_cleaned[features_original_accident[11:14]].astype(int)  # Changing bool to int
data_accident_cleaned[features_original_accident[34]] = data_accident_cleaned[features_original_accident[34]].astype(dtype='Int64')  # change weekday to int
to_int(data_accident_cleaned, features_original_accident, 1, 2)  # type
to_int(data_accident_cleaned, features_original_accident, 6, 2)  # severity
to_int(data_accident_cleaned, features_original_accident, 14, 4)  # road type
to_int(data_accident_cleaned, features_original_accident, 29, 3)  # week day
data_accident_cleaned = find_day(data_accident_cleaned)
data_accident_cleaned.to_pickle("tidy_data/RoadTrafficAccidentLocations_cleaned.pickle")
data_accident_cleaned.to_csv("tidy_data/RoadTrafficAccidentLocations_cleaned.csv")
"""

# To read the already generated accident tidy_data data uncomment the following line
# data_accident_cleaned = pd.read_pickle("tidy_data/RoadTrafficAccidentLocations_cleaned.pickle")

# =============================================================================
# clean meteo data
# to generate all meteo tidy_data data, uncomment this section
"""
data_meteo_cleaned = meteo_date_prep(data_meteo)  # Create new df with temperature
data_meteo_cleaned.to_pickle("tidy_data/ugz_ogd_meteo_h1_2011-2020_cleaned.pickle")
data_meteo_cleaned.to_csv("tidy_data/ugz_ogd_meteo_h1_2011-2020_cleaned.csv")
"""

# To read the already generated meteo tidy_data data uncomment the following line
# data_meteo_cleaned = pd.read_pickle("tidy_data/ugz_ogd_meteo_h1_2011-2020_cleaned.pickle")  # Load the meteo df

# =============================================================================
# clean biker and pedestrian counter data
# velo_fuss_date_prep(data_velo_fussgang11).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang12).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2012_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang13).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2013_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang14).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2014_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang15).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2015_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang16).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2016_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang17).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2017_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang18).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2018_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang19).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2019_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang20).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")

# =============================================================================
# changing date format
# data_velo_fussgang11_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang12_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2012_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang13_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2013_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang14_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2014_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang15_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2015_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang16_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2016_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang17_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2017_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang18_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2018_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang19_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2019_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# data_velo_fussgang20_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")


data_velo_fussgang11_c = pd.read_pickle("tidy_data/pre_tidy_fussgaenger_velo/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.pickle")
df = pd.read_csv("tidy_data/RoadTrafficAccidentLocations_cleaned.csv")
d = associate_coord_to_accident_coord(200, df, data_velo_fussgang11_c)
d.to_csv("tidy_data/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_merge_ready_200.csv")
d.to_pickle("tidy_data/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_merge_ready_200.pickle")

# data_auto_c = pd.read_csv("tidy_data/pre_tidy_auto/sid_dav_verkehrszaehlung_miv_OD2031_2012-2020.csv")
# df = pd.read_csv("tidy_data/RoadTrafficAccidentLocations_cleaned.csv")
# d = associate_coord_to_accident_coord_cars(1000, df, data_auto_c)
# d.to_csv("tidy_data/2011-2020_verkehrszaehlungen_werte_auto_merge_ready_1000.csv")
# d.to_pickle("tidy_data/2011-2020_verkehrszaehlungen_werte_auto_merge_ready_1000.pickle")


# # TODO: mach en for loop und append alles in e liste und denn speichere alles i eim df wo als csv und pickle speicherisch...
# data_counting_cleaned = pd.read_csv("tidy_data/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_counting_cleaned.dropna(inplace=True)
# data_counting_cleaned.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_counting_cleaned = find_day(data_counting_cleaned)

# =============================================================================
# clean auto count data
# data_auto_12 = pd.read_csv("raw_data/Verkehrszahelung_Autos/sid_dav_verkehrszaehlung_miv_OD2031_2012.csv")



# =============================================================================
# merge dataframes
# data_merged = pd.merge(data_accident_cleaned, data_meteo_cleaned, how='left', right_index=True, left_index=True)
# data_merged.dropna(inplace=True)
# data_merged.to_pickle("tidy_data/data_merged.pickle")
# data_merged.to_csv("tidy_data/data_merged.csv")
data_velo_fussgang_cleaned = pd.read_pickle("tidy_data/2011-2020_verkehrszaehlungen_werte_fussgaenger_velo_merge_ready.pickle")
data_merged = pd.read_pickle("tidy_data/data_merged.pickle")
data_merged = pd.merge(data_merged, data_velo_fussgang_cleaned, how='left', on=['Date', 'AccidentLocation_CHLV95_E',
                            'AccidentLocation_CHLV95_N'])  # The key of the datetime index is 'Date'!



# =============================================================================
