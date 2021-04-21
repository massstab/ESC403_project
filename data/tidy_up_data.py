# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:30:42 2021
@author: marszpd
"""

import pandas as pd
import numpy as np

# =============================================================================
# Raw data
# =============================================================================
data_accident = pd.read_csv("raw/RoadTrafficAccidentLocations.csv")
data_meteo = pd.read_csv("raw/ugz_ogd_meteo_h1_2011.csv")

features_original_accident = ['AccidentUID', 'AccidentType', 'AccidentType_de',
                     'AccidentType_fr','AccidentType_it', 'AccidentType_en',
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
# Tidy data functions
# =============================================================================
def to_int(data, feature_list, m, k):
    """
    Removes unwanted string part of the data.

    Parameters
    ----------
    data : Data Frame
        Pandas data frame.
    feature_list : list
        List of features.
    m : int
        Feature number, as saved in the list feature_list.
    k : int
        Length of the unnecessary string.

    Returns
    -------
    None.

    """

    integerfy = lambda lst, i: [int(item[i:]) for item in lst]
    data_arr = data[feature_list[m]].to_numpy()
    temp = integerfy(data_arr, k)
    data[feature_list[m]] = temp  # make str of form "str" + "number" to number

def meteo_date_prep(df):
    """
    Brings the meteo raw data in the right format and computes the average temperature from the two measuring stations
    :param df: raw meteo dataframe
    :return: the new meteo dataframe
    """
    new_df = pd.DataFrame(columns=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour', 'AvgTemperature'])
    i = 0
    for row in df[['Datum', 'Standort', 'Parameter', 'Wert']].values:
        if row[2] == 'T':
            if row[1] == "Zch_Schimmelstrasse":
                summand = row[3]
                year, month, day, hour = row[0][:4], row[0][5:7], row[0][8:10], row[0][11:13]
                date = pd.to_datetime(year + "-" + month + "-" + day + "-" + hour)
                weekday = date.weekday() + 1  # 1,...,7 Monday,...,Sunday
                new_df.at[i, 'AccidentYear'] = year
                new_df.at[i, 'AccidentMonth'] = month
                new_df.at[i, 'AccidentWeekDay'] = weekday
                new_df.at[i, 'AccidentHour'] = hour
            else:
                new_df.at[i, 'AvgTemperature'] = 0.5 * (row[3] + summand)
                i += 1
    return new_df

# =============================================================================
# Perform cleaning
# =============================================================================
# clean accident data
data_accident_cleaned = data_accident[features_croped_accident].copy()  # complete new df with only used features
nan_index = data_accident_cleaned[data_accident_cleaned.isin([np.nan, np.inf, -np.inf]).any(1)].index  # get indices from nan values

data_accident_cleaned.drop(nan_index, inplace=True)  # drop nan values
data_accident_cleaned[features_original_accident[11:14]] = data_accident_cleaned[features_original_accident[11:14]].astype(int)  # Changing bool to int
data_accident_cleaned[features_original_accident[34]] = data_accident_cleaned[features_original_accident[34]].astype(int)  # change weekday to int
to_int(data_accident_cleaned, features_original_accident, 1, 2)  # type
to_int(data_accident_cleaned, features_original_accident, 6, 2)  # severity
to_int(data_accident_cleaned, features_original_accident, 14, 4)  # road type
to_int(data_accident_cleaned, features_original_accident, 29, 3)  # week day

# clean meteo data
# data_meteo_cleaned = meteo_date_prep(data_meteo)  # Create new df with temperature
# data_meteo_cleaned.to_pickle("tidy/temp_meteo.pickle")  # Save the df for faster load
data_accident_cleaned = pd.read_pickle("tidy/temp_meteo.pickle")  # Load the meteo df


data_accident_cleaned.to_csv("tidy/RoadTrafficAccidentLocations_cleaned.csv")
# =============================================================================
