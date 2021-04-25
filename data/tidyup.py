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

    integerfy = lambda lst, i: [int(item[i:]) for item in lst]
    data_arr = data[feature_list[m]].to_numpy()
    temp = integerfy(data_arr, k)
    data[feature_list[m]] = temp  # make str of form "str" + "number" to number


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
        raw meteo dataframe

    Returns
    -------
    The bike and pedestrain dataframe
    """
    new_df = pd.DataFrame(columns=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay',
               'AccidentHour','AccidentLocation_CHLV95_E','AccidentLocation_CHLV95_N',
               'SumBikerNumber', 'SumPedastrianNumber'])
    i = 0
    id_lst = list(set(df['FK_STANDORT'].to_numpy())) # to be pedantic, it's not the id but the location id
    for i, id_number in enumerate(id_lst):
        data_i = df[df['FK_STANDORT'] == id_number].values # gives a dataframe with just the data from that specific id_number
        number_data_points = data_i.shape[0]
        for j in range(int(number_data_points/4 - 1)): # division by four due to summation

            # set date as done by the function meteo_date_prep
            year, month, day, hour = data_i[j][2][:4], data_i[j][2][5:7], data_i[j][2][8:10], data_i[j][2][11:13]
            date = pd.to_datetime(year + "-" + month + "-" + day + "-" + hour)
            weekday = date.weekday() + 1  # 1,...,7 Monday,...,Sunday
            new_df.at[(i + 1)*j, 'AccidentYear'] = int(year)
            new_df.at[(i + 1)*j, 'AccidentMonth'] = int(month)
            new_df.at[(i + 1)*j, 'AccidentWeekDay'] = int(weekday)  # Couldń't exctract which day of the month that is...
            new_df.at[(i + 1)*j, 'AccidentHour'] = int(hour)

            # set coordinates
            new_df.at[(i + 1)*j, 'AccidentLocation_CHLV95_E'] = data_i[0][7]
            new_df.at[(i + 1)*j, 'AccidentLocation_CHLV95_N'] = data_i[0][8]

            # set sum of the data
            new_df.at[(i + 1)*j, 'SumBikerNumber'] = __helper_velo_fuss(data_i[j*4:(j+1)*4][:,3], data_i[j*4:(j+1)*4][:,4])
            new_df.at[(i + 1)*j, 'SumPedastrianNumber'] = __helper_velo_fuss(data_i[j*4:(j+1)*4][:,5], data_i[j*4:(j+1)*4][:,6])

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
        if red1 == [] and 0 < len(red2) <= 4:
            return sum(red2)
        elif red2 == [] and 0 < len(red1) <= 4:
            return sum(red1)
        else:
            return 0.5 * sum(red1 + red2)

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
data_velo_fussgang19 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2019_verkehrszaehlungen_werte_fussgaenger_velo.csv")
data_velo_fussgang20 = pd.read_csv("raw_data/Verkehrszaehlungen_werte_fussgaenger/2020_verkehrszaehlungen_werte_fussgaenger_velo.csv")

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
# velo_fuss_date_prep(data_velo_fussgang11).to_csv("tidy/2011_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang12).to_csv("tidy/2012_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang13).to_csv("tidy/2013_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang14).to_csv("tidy/2014_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang15).to_csv("tidy/2015_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang16).to_csv("tidy/2016_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang17).to_csv("tidy/2017_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
# velo_fuss_date_prep(data_velo_fussgang18).to_csv("tidy/2018_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
velo_fuss_date_prep(data_velo_fussgang19).to_csv("tidy/2019_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")
velo_fuss_date_prep(data_velo_fussgang20).to_csv("tidy/2020_verkehrszaehlungen_werte_fussgaenger_velo_cleaned.csv")

# =============================================================================
# merge dataframes
# data_merged = pd.merge(data_accident_cleaned, data_meteo_cleaned, how='left', right_index=True, left_index=True)
# data_merged.dropna(inplace=True)
# data_merged.to_pickle("tidy_data/data_merged.pickle")
# data_merged.to_csv("tidy_data/data_merged.csv")

# =============================================================================
