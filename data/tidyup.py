# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:30:42 2021
@author: marszpd, dtm
"""
import numpy as np
import pandas as pd


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


# !!! please do not alter the format of the docstrings,
# in this format they will be displayed nicely in spyder
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
    new_df = pd.DataFrame(columns=['Date', 'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N',
                                   'SumBikerNumber', 'SumPedastrianNumber'])
    i = 0
    id_lst = list(set(df['FK_STANDORT'].to_numpy())) # to be pedantic, it's not the id but the location id
    for i, id_number in enumerate(id_lst):
        data_i = df[df['FK_STANDORT'] == id_number].values # gives a dataframe with just the data from that specific id_number
        number_data_points = data_i.shape[0]
        for j in range(int(number_data_points/4 - 1)): # division by four due to summation

            # set date as done by the function meteo_date_prep, well at least as done by the old version
            year, month, day, hour = data_i[j][2][:4], data_i[j][2][5:7], data_i[j][2][8:10], data_i[j][2][11:13]
            date = pd.to_datetime(year + "-" + month + "-" + day + "-" + hour)
            new_df.at[(i + 1)*j, 'Date'] = date
            # weekday = date.weekday() + 1  # 1,...,7 Monday,...,Sunday
            # new_df.at[(i + 1)*j, 'AccidentYear'] = int(year)
            # new_df.at[(i + 1)*j, 'AccidentMonth'] = int(month)
            # new_df.at[(i + 1)*j, 'AccidentWeekDay'] = int(weekday)  # Couldń't exctract which day of the month that is...
            # new_df.at[(i + 1)*j, 'AccidentHour'] = int(hour)

            # set coordinates
            new_df.at[(i + 1)*j, 'AccidentLocation_CHLV95_E'] = data_i[0][7]
            new_df.at[(i + 1)*j, 'AccidentLocation_CHLV95_N'] = data_i[0][8]

            # set sum of the data
            new_df.at[(i + 1)*j, 'SumBikerNumber'] = __helper_velo_fuss(data_i[j*4:(j+1)*4][:,3], data_i[j*4:(j+1)*4][:,4])
            new_df.at[(i + 1)*j, 'SumPedastrianNumber'] = __helper_velo_fuss(data_i[j*4:(j+1)*4][:,5], data_i[j*4:(j+1)*4][:,6])
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
        if red1 == [] and 0 < len(red2) <= 4: # it turns out that the inequality i.e. <= isn't neccessary since one can see that once one value is nan all 3 others are too
            return sum(red2)
        elif red2 == [] and 0 < len(red1) <= 4: # it turns out that the inequality i.e. <= isn't neccessary since one can see that once one value is nan all 3 others are too
            return sum(red1)
        else:
            return 0.5 * sum(red1 + red2)


def auto_prep(df):
    pass


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
    "Merged" DataFrame.
    """

    # find the unique coordinates
    unique_coord_df1 = df1.groupby(['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N']).count().index
    unique_coord_df2 = df2.groupby(['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N']).count().index

    df1_new = df1.copy()
    df1_new['SumBikerNumber'] = np.nan
    df1_new['SumPedastrianNumber'] = np.nan

    # find the nearest neighbours
    for i, coord1 in enumerate(unique_coord_df1):
        x1, y1 = coord1
        temp1, temp2 = [], []
        for coord2 in unique_coord_df2:
            x2, y2 = coord2
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if len(temp2) == 0:
                temp2.append(distance)
            if distance <= radius and distance == temp2[-1]:
                print("sos, indeed you should indicate an average")
                temp1.append(coord2)
            elif distance <= radius and distance < temp2[-1]:
                temp1 = []
                temp1.append(coord2)
                temp2[-1] = distance

        # perform association
        indices1 = df1[df1[['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N']] == coord1].dropna(axis=0).index

        # caution here it is assumed that the dates for all coords in temp1 are the same, (which should hold, just to clarify the made choices here)
        indices2 = df2[df2[['AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N']] == temp1[-1]].dropna(axis=0).index
        dates_df2 = df2['Date'].iloc[indices2].values

        for i in indices1:
            aviable_date = df1_new['Date'].iloc[i]
            if aviable_date in dates_df2:

                temp3 = df2[df2['Date'] == aviable_date]
                value_bike = temp3['SumBikerNumber'].values
                value_ped = temp3['SumPedastrianNumber'].values

                # drop nans
                value_bike = value_bike[np.logical_not(np.isnan(value_bike))]
                value_ped = value_ped[np.logical_not(np.isnan(value_ped))]

                if not len(value_bike):
                    value_bike = np.nan # if empty assign nan
                else:
                    value_bike = value_bike.sum()/len(value_bike) # take the average if there are multiple points with the same distance

                if not len(value_ped):
                    value_ped = np.nan  # if empty assign nan
                else:
                    value_ped = value_ped.sum()/len(value_ped) # take the average if there are multiple points with the same distance

                df1_new['SumBikerNumber'].iloc[i] = value_bike
                df1_new['SumPedastrianNumber'].iloc[i] = value_ped

    return df1_new


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
data_velo_fussgang11 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2011_verkehrszaehlungen_werte_fussgaenger_velo.csv")
data_velo_fussgang12 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2012_verkehrszaehlungen_werte_fussgaenger_velo.csv")
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
velo_fuss_date_prep(data_velo_fussgang11).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
velo_fuss_date_prep(data_velo_fussgang12).to_csv("tidy_data/pre_tidy_fussgaenger_velo/2012_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
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
# data_velo_fussgang11_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang12_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2012_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang13_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2013_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang14_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2014_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang15_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2015_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang16_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2016_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang17_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2017_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang18_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2018_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang19_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2019_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_velo_fussgang20_c = pd.read_csv("tidy_data/pre_tidy_fussgaenger_velo/2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)

# data_velo_fussgang11_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang12_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang13_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang14_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang15_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang16_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang17_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang18_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang19_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_velo_fussgang20_c.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)

# find_day(data_velo_fussgang11_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang12_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2012_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang13_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2013_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang14_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2014_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang15_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2015_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang16_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2016_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang17_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2017_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang18_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2018_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang19_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2019_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# find_day(data_velo_fussgang20_c).to_csv("tidy_data/pre_tidy_fussgaenger_velo/final_tidy/2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")


# TODO: mach en for loop und append alles in e liste und denn speichere alles i eim df wo als csv und pickle speicherisch...
# data_counting_cleaned = pd.read_csv("tidy_data/2012_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv", index_col=0)
# data_counting_cleaned.dropna(inplace=True)
# data_counting_cleaned.sort_values(by=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'], axis=0, inplace=True, ignore_index=True)
# data_counting_cleaned = find_day(data_counting_cleaned)
# print(data_counting_cleaned.head())


# =============================================================================
# merge dataframes
# data_merged = pd.merge(data_accident_cleaned, data_meteo_cleaned, how='left', right_index=True, left_index=True)
# data_merged.dropna(inplace=True)
# data_merged.to_pickle("tidy_data/data_merged.pickle")
# data_merged.to_csv("tidy_data/data_merged.csv")

# =============================================================================
