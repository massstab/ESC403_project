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
    df and computes the average temperature from the two measuring stations
    :param df: raw_data meteo dataframe
    :return: the new meteo dataframe
    """
    new_df = pd.DataFrame(columns=['Date', 'AvgTemperature'])
    i = 0
    temp_sum = 0
    visited = False
    num_of_stations = 0
    current_hour = '00'
    first_station = df['Standort'].values[0]
    start_time = df['Datum'].values[0][11:13]
    if start_time != current_hour:
        print('Please start the data frame at time 00:00')
    for row in df[['Datum', 'Standort', 'Parameter', 'Wert']].values:
        if row[0][11:13] != current_hour:
            new_df.at[i, 'AvgTemperature'] = round(temp_sum / num_of_stations, 1)
            i += 1  # Update index for the next entry in new_df
            temp_sum = 0
            visited = False
            num_of_stations = 0
            current_hour = row[0][11:13]
            first_station = row[1]
            current_station = first_station
        if (row[2] == 'T'):
            if not visited:
                visited = True
                num_of_stations += 1
                temp_sum += row[3]
                year, month, day, hour = row[0][:4], row[0][5:7], row[0][8:10], row[0][11:13]
                date = pd.to_datetime(year + "-" + month + "-" + day + "-" + hour)
                new_df.at[i, 'Date'] = date
                print(date.year, date.month_name())
                continue

            current_station = row[1]
            if current_station != first_station:
                num_of_stations += 1
                temp_sum += row[3]
                continue

    new_df.at[i, 'AvgTemperature'] = round(temp_sum / num_of_stations, 1)  # The last entry
    new_df.set_index('Date', inplace=True)
    return new_df

# =============================================================================
# Raw data
# =============================================================================
data_accident = pd.read_csv("raw_data/RoadTrafficAccidentLocations.csv")
data_meteo = pd.read_csv("raw_data/ugz_ogd_meteo_h1_2011-2020.csv")

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

features_meteo = ['Datum', 'Standort', 'Parameter', 'Intervall', 'Einheit',
                  'Wert', 'Status']

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
data_accident_cleaned = pd.read_pickle("tidy_data/RoadTrafficAccidentLocations_cleaned.pickle")

# =============================================================================
# clean meteo data
# to generate all meteo tidy_data data, uncomment this section
"""
data_meteo_cleaned = meteo_date_prep(data_meteo)  # Create new df with temperature
data_meteo_cleaned.to_pickle("tidy_data/ugz_ogd_meteo_h1_2011-2020_cleaned.pickle")
data_meteo_cleaned.to_csv("tidy_data/ugz_ogd_meteo_h1_2011-2020_cleaned.csv")
"""

# To read the already generated meteo tidy_data data uncomment the following line
data_meteo_cleaned = pd.read_pickle("tidy_data/ugz_ogd_meteo_h1_2011-2020_cleaned.pickle")  # Load the meteo df

# =============================================================================
# merge dataframes
data_merged = pd.merge(data_accident_cleaned, data_meteo_cleaned, how='left', right_index=True, left_index=True)
data_merged.dropna(inplace=True)
data_merged.to_pickle("tidy_data/data_merged.pickle")
data_merged.to_csv("tidy_data/data_merged.csv")


# =============================================================================