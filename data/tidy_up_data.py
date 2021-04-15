# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:30:42 2021
@author: marszpd
"""

import pandas as pd

# =============================================================================
# Raw data
# =============================================================================
data_accident = pd.read_csv("raw/RoadTrafficAccidentLocations.csv")
data_meteo = pd.read_csv("raw/ugz_ogd_meteo_h1_2011.csv")

features_accident = ['AccidentUID', 'AccidentType', 'AccidentType_en',
                'AccidentSeverityCategory', 'AccidentSeverityCategory_en',
                'AccidentInvolvingPedestrian', 'AccidentInvolvingBicycle',
                'AccidentInvolvingMotorcycle', 'RoadType', 'RoadType_en',
                'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N',
                'CantonCode', 'MunicipalityCode', 'AccidentYear',
                'AccidentMonth','AccidentMonth_en', 'AccidentWeekDay',
                'AccidentWeekDay_en', 'AccidentHour', 'AccidentHour_text']

features_meteo = ['Datum', 'Standort', 'Parameter', 'Intervall', 'Einheit', 'Wert',
                  'Status']

# =============================================================================
# Tidy data
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

    integerfy = lambda lst, i: [int(item[i:]) for item in  lst]
    print(data[['AccidentYear','AccidentMonth', 'AccidentMonth_de']])
    print(data['AccidentYear']/10)
    data_arr = data[feature_list[m]].to_numpy()
    temp = integerfy(data_arr, k)
    data[feature_list[m]] = temp # make str of form "at" + "number" to integers number


to_int(data_accident, features_accident, 1, 2)

# to_int(data_accident, features_accident, 3, 2)

# print(data_accident[features_accident[8:10]])
# data_tidy =





