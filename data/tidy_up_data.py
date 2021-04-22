# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:30:42 2021
@author: marszpd, dtm
"""

import pandas as pd
import numpy as np

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
                new_df.at[i, 'AccidentYear'] = int(year)
                new_df.at[i, 'AccidentMonth'] = int(month)
                new_df.at[i, 'AccidentWeekDay'] = int(weekday)  # Couldń't exctract which day of the month that is...
                new_df.at[i, 'AccidentHour'] = int(hour)
            else:
                new_df.at[i, 'AvgTemperature'] = round(0.5 * (row[3] + summand), 1)
                i += 1  # Update index for the next entry in new_df
    return new_df

# =============================================================================
# Raw data
# =============================================================================
data_accident = pd.read_csv("raw/RoadTrafficAccidentLocations.csv")
data_meteo11 = pd.read_csv("raw/ugz_ogd_meteo_h1_2011.csv")
data_meteo12 = pd.read_csv("raw/ugz_ogd_meteo_h1_2012.csv")
data_meteo13 = pd.read_csv("raw/ugz_ogd_meteo_h1_2013.csv")
data_meteo14 = pd.read_csv("raw/ugz_ogd_meteo_h1_2014.csv")
data_meteo15 = pd.read_csv("raw/ugz_ogd_meteo_h1_2015.csv")
data_meteo16 = pd.read_csv("raw/ugz_ogd_meteo_h1_2016.csv")
data_meteo17 = pd.read_csv("raw/ugz_ogd_meteo_h1_2017.csv")
data_meteo18 = pd.read_csv("raw/ugz_ogd_meteo_h1_2018.csv")
data_meteo19 = pd.read_csv("raw/ugz_ogd_meteo_h1_2019.csv")
data_meteo20 = pd.read_csv("raw/ugz_ogd_meteo_h1_2020.csv")

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

# Don't change this list because it is used by the
features_croped_meteo = ['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour', 'AvgTemperature']

# =============================================================================
# Perform cleaning
# =============================================================================
# clean accident data
# data_accident_cleaned = data_accident[features_croped_accident].copy()  # complete new df with only used features
# nan_index = data_accident_cleaned[data_accident_cleaned.isin([np.nan, np.inf, -np.inf]).any(1)].index  # get indices from nan values

# data_accident_cleaned.drop(nan_index, inplace=True)  # drop nan values
# data_accident_cleaned[features_original_accident[11:14]] = data_accident_cleaned[features_original_accident[11:14]].astype(int)  # Changing bool to int
# data_accident_cleaned[features_original_accident[34]] = data_accident_cleaned[features_original_accident[34]].astype(int)  # change weekday to int
# to_int(data_accident_cleaned, features_original_accident, 1, 2)  # type
# to_int(data_accident_cleaned, features_original_accident, 6, 2)  # severity
# to_int(data_accident_cleaned, features_original_accident, 14, 4)  # road type
# to_int(data_accident_cleaned, features_original_accident, 29, 3)  # week day

# data_accident_cleaned.to_csv("tidy/RoadTrafficAccidentLocations_cleaned.csv")

# =============================================================================
# clean meteo data
# data_meteo_cleaned11 = meteo_date_prep(data_meteo11)  # Create new df with temperature
# data_meteo_cleaned12 = meteo_date_prep(data_meteo12)  # Create new df with temperature
# data_meteo_cleaned13 = meteo_date_prep(data_meteo13)  # Create new df with temperature
data_meteo_cleaned14 = meteo_date_prep(data_meteo14)  # Create new df with temperature
# data_meteo_cleaned15 = meteo_date_prep(data_meteo15)  # Create new df with temperature
# data_meteo_cleaned16 = meteo_date_prep(data_meteo16)  # Create new df with temperature
# data_meteo_cleaned17 = meteo_date_prep(data_meteo17)  # Create new df with temperature
# data_meteo_cleaned18 = meteo_date_prep(data_meteo18)  # Create new df with temperature
# data_meteo_cleaned19 = meteo_date_prep(data_meteo19)  # Create new df with temperature
# data_meteo_cleaned20 = meteo_date_prep(data_meteo20)  # Create new df with temperature

# data_meteo_cleaned11 = pd.read_pickle("tidy/temp_meteo11.pickle")  # Load the meteo df
# data_meteo_cleaned12 = pd.read_pickle("tidy/temp_meteo12.pickle")  # Load the meteo df
# data_meteo_cleaned13 = pd.read_pickle("tidy/temp_meteo13.pickle")  # Load the meteo df
# data_meteo_cleaned14 = pd.read_pickle("tidy/temp_meteo14.pickle")  # Load the meteo df
# data_meteo_cleaned15 = pd.read_pickle("tidy/temp_meteo15.pickle")  # Load the meteo df
# data_meteo_cleaned16 = pd.read_pickle("tidy/temp_meteo16.pickle")  # Load the meteo df
# data_meteo_cleaned17 = pd.read_pickle("tidy/temp_meteo17.pickle")  # Load the meteo df
# data_meteo_cleaned18 = pd.read_pickle("tidy/temp_meteo18.pickle")  # Load the meteo df
# data_meteo_cleaned19 = pd.read_pickle("tidy/temp_meteo19.pickle")  # Load the meteo df
# data_meteo_cleaned20 = pd.read_pickle("tidy/temp_meteo20.pickle")  # Load the meteo df
#
# data_meteo_cleaned11.to_csv("tidy/ugz_ogd_meteo_h1_2011_cleaned.csv")
# data_meteo_cleaned12.to_csv("tidy/ugz_ogd_meteo_h1_2012_cleaned.csv")
# data_meteo_cleaned13.to_csv("tidy/ugz_ogd_meteo_h1_2013_cleaned.csv")
# data_meteo_cleaned14.to_csv("tidy/ugz_ogd_meteo_h1_2014_cleaned.csv")
# data_meteo_cleaned15.to_csv("tidy/ugz_ogd_meteo_h1_2015_cleaned.csv")
# data_meteo_cleaned16.to_csv("tidy/ugz_ogd_meteo_h1_2016_cleaned.csv")
# data_meteo_cleaned17.to_csv("tidy/ugz_ogd_meteo_h1_2017_cleaned.csv")
# data_meteo_cleaned18.to_csv("tidy/ugz_ogd_meteo_h1_2018_cleaned.csv")
# data_meteo_cleaned19.to_csv("tidy/ugz_ogd_meteo_h1_2019_cleaned.csv")
# data_meteo_cleaned20.to_csv("tidy/ugz_ogd_meteo_h1_2020_cleaned.csv")

# # merge dataframes
# data_merged = pd.merge(data_accident_cleaned, data_meteo_cleaned, how='left', on=['AccidentYear', 'AccidentMonth', 'AccidentWeekDay', 'AccidentHour'])

# =============================================================================
