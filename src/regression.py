# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:17:45 2021
@author: dtm, marszpd
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import data_all as df
import statsmodels.formula.api as sm

# =============================================================================

pd.options.display.max_columns = 5
pd.options.display.max_rows = 10
pd.set_option('display.width', 150)

features = ['Date', 'AccidentType', 'AccidentSeverityCategory', 'AccidentInvolvingPedestrian',
            'AccidentInvolvingBicycle', 'AccidentInvolvingMotorcycle', 'RoadType',
            'AccidentLocation_CHLV95_E', 'AccidentLocation_CHLV95_N', 'AvgTemperature',
            'AvgRainDur', 'SumBikerNumber', 'SumPedastrianNumber', 'SumCars']

# =============================================================================
def last_day_of_month(any_day):
    """Gives last day of month. See reference [1].

    Parameter
    ---------
    any_day: daytime object
        Date as daytime object.

    Reference
    ---------
    .. [1] Anwsered by user augustomen, (2021, Mai 14), How to get the last day
       of the month?, https://stackoverflow.com/a/13565185
    """
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)


def sum_per_time(df, per_hour=False, per_day=False, per_month=False, per_year=True,
                 x_axis='AvgTemperature', y_axis='SumAccidents'):
    """ Calculates the sum of accidents within the given time interval.

    Parameters
    ----------
    df : pandas dataframe
        DataFarme.
    per_hour : boolean, optional
        If True all accidents occuring during the given destinct hours will be
        summed. The default is False. (description wonky)
    per_day : boolean, optional
        If True all accidents occuring during the given destinct day will be
        summed. The default is False. (description wonky)
    per_year : boolean, optional
        If True all accidents occuring during the given destinct year will be
        summed. The default is False. (description wonky)

    Return
    ------
    DataFrame with summed accidents.
    """

    df['Date'] = pd.to_datetime(df['Date'])

    if per_hour:
        unique_date_temp = df.groupby(['Date'])
        unique_date = unique_date_temp['Date'].count().to_frame()

        df_new = df.drop_duplicates(subset=['Date', x_axis])
        unique_date.insert(1, x_axis, df_new[x_axis].to_numpy())
        unique_date.rename(columns={'Date': 'SumAccidents'}, inplace=True)

    if per_day:
        df_new = df[['Date', x_axis]].copy()
        df_new.insert(2, 'Day', df_new['Date'].dt.day.to_numpy())
        df_new.insert(3, 'Month', df_new['Date'].dt.month.to_numpy())
        df_new.insert(4, 'Year', df_new['Date'].dt.year.to_numpy())

        unique_date_temp = df_new.groupby(['Year', 'Month', 'Day'])
        unique_date = unique_date_temp[x_axis].count().to_frame()
        unique_date.rename(columns={x_axis: 'SumAccidents'}, inplace=True)

        unique_date.reset_index(level=['Year', 'Month', 'Day'], inplace=True)
        unique_date['Date'] = pd.to_datetime(unique_date[['Day','Month','Year']])
        unique_date.drop(['Day','Month','Year'], axis = 1, inplace=True)
        unique_date.insert(2, x_axis, unique_date_temp[x_axis].mean().to_numpy())


    if per_month:
        df_new = df[['Date', x_axis]].copy()
        df_new.insert(2, 'Day', df_new['Date'].dt.day.to_numpy())
        df_new.insert(3, 'Month', df_new['Date'].dt.month.to_numpy())
        df_new.insert(4, 'Year', df_new['Date'].dt.year.to_numpy())

        unique_date_temp = df_new.groupby(['Year', 'Month'])
        unique_date = unique_date_temp[x_axis].count().to_frame()
        unique_date.rename(columns={x_axis: 'SumAccidents'}, inplace=True)

        unique_date.reset_index(level=['Year', 'Month'], inplace=True)
        lst_last_day_month = [last_day_of_month(datetime.date(year, unique_date['Month'].to_numpy()[i], 1)).day for i, year in enumerate(unique_date['Year'].to_numpy())]
        new = dict(day=lst_last_day_month)
        rnm = dict(year='Year', month='Month')
        dates_temp = unique_date[['Year', 'Month']].assign(Date=pd.to_datetime(unique_date[['Year', 'Month']].rename(columns=rnm).assign(**new)))

        unique_date.drop(['Year','Month'], axis = 1, inplace=True)
        unique_date.insert(1, 'Date', dates_temp['Date'].to_numpy())
        unique_date.insert(2, x_axis, unique_date_temp[x_axis].mean().to_numpy())


    if per_year:
        df_new = df[['Date', x_axis]].copy()
        df_new.insert(2, 'Day', df_new['Date'].dt.day.to_numpy())
        df_new.insert(3, 'Month', df_new['Date'].dt.month.to_numpy())
        df_new.insert(4, 'Year', df_new['Date'].dt.year.to_numpy())

        unique_date_temp = df_new.groupby(['Year'])
        unique_date = unique_date_temp[x_axis].count().to_frame()
        unique_date.rename(columns={x_axis: 'SumAccidents'}, inplace=True)

        unique_date.reset_index(level=['Year'], inplace=True)
        new = dict(month=12, day=31)
        rnm = dict(year='Year', month='Month')
        dates_temp = unique_date[['Year']].assign(Date=pd.to_datetime(unique_date[['Year']].rename(columns=rnm).assign(**new)))

        unique_date.drop(['Year'], axis = 1, inplace=True)
        unique_date.insert(1, 'Date', dates_temp['Date'].to_numpy())
        unique_date.insert(2, x_axis, unique_date_temp[x_axis].mean().to_numpy())

    return unique_date

# =============================================================================

if __name__ == "__main__":

    lst = np.identity(4)

    for item in lst:
        # df = df[(df['AccidentSeverityCategory'] == 2)]
        z_axis = "Date, arbitrary label"
        y_axis = 'SumAccidents'
        x_axis = 'AvgRainDur'

        # df = df[(df['AccidentSeverityCategory'] == 1)]
        df_sum = sum_per_time(df, per_hour=item[0], per_day=item[1], per_month=item[2],
                              per_year=item[3], x_axis=x_axis, y_axis=y_axis)
        z = list(range(len(df_sum.index.values)))
        x = df_sum[x_axis].values
        y = df_sum[y_axis].values
        print(df_sum.keys())

        # collect them in a pandas dfframe
        df_sum = pd.DataFrame({'Date': z, x_axis: x, y_axis: y})

        # linear regression fit111
        reg = sm.ols(formula='z ~ x + y', data=df_sum).fit()
        print(reg.summary())

        from mpl_toolkits.mplot3d import Axes3D
        a0, a1, a2 = reg.params
        print('The fitting formula is: z = {0:} + {1:} x + {2:} y'.format(round(a0, 3),
                                                                          round(a1, 3),
                                                                          round(a2, 3)))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=1)

        # set up the surface plot
        X = np.linspace(min(x), max(x), 100)
        Y = np.linspace(min(y), max(y), 100)
        XX, YY = np.meshgrid(X, Y)
        ZZ = a0 + a1 * XX + a2 * YY
        ax.plot_surface(XX, YY, ZZ, alpha=0.3, color='red')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)

        plt.show()
