# -*- coding: utf-8 -*-
"""
Created on 13 May 2021
@author: dtm
according to this tutorial: https://www.tensorflow.org/tutorials/structured_data/feature_columns
"""

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split
from datasets import data_all as dataframe
from helpers import df_to_dataset

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# set some pandas option to use the whole terminal width to display results
pd.options.display.max_columns = 10
pd.set_option('display.width', 200)

dataframe = dataframe.drop(columns=['Date', 'SumBikerNumber',
                                    'SumBikerNumber', 'SumCars', 'SumPedestrianNumber'])
# Shorten feature names just for convenient output format
new_cols = {"AccidentSeverityCategory": "Severity", "AccidentType": "AccType", "AvgTemperature": "Temperature",
            "AvgRainDur": "RainDur",
            "AccidentInvolvingPedestrian": "Pedestrian", "AccidentInvolvingBicycle": "Bicycle",
            "AccidentInvolvingMotorcycle": "Motorcycle", "AccidentLocation_CHLV95_E": "LocationE",
            "AccidentLocation_CHLV95_N": "LocationN"}
dataframe.rename(columns=new_cols, inplace=True)
dataframe.loc[dataframe['RoadType'] == 9, 'RoadType'] = 5

sequential_model = False
decision_tree = True
random_forest = False


if sequential_model:
    target = 'RoadType'
    dataframe['target'] = dataframe[target].values

    # Split into train test and validation
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    batch_size = 1024
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    feature_columns = []
    # Define the feature columns
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType','LocationE', 'LocationN', 'Temperature', 'RainDur']
    for header in ['Severity', 'Bicycle', 'Motorcycle', 'AccType', 'Temperature']:
        feature_columns.append(feature_column.numeric_column(header))

    # Use DenseFeatures layer as an input to the Keras model
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # Take a look at a batch of a feature
    # for feature_batch, label_batch in train_ds.take(1):
    #     print('Every feature:', list(feature_batch.keys()))
    #     print('A batch of RoadType:', feature_batch['RoadType'])
    #     print('A batch of Bicycle:', feature_batch['Bicycle'])
    #     print('A batch of Motorcycle:', feature_batch['Motorcycle'])
    #     print('A batch of Temperature:', feature_batch['Temperature'])
    #     print('A batch of targets:', label_batch)

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(6, activation='softmax')
    ])

    optimizer = optimizers.Adam(lr=0.01)  # Defines the learning rate
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

    history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=None)
    loss, accuracy = model.evaluate(test_ds)
    # print("Accuracy", accuracy)
    # print(dataframe.info())
    # pred = model.predict_classes(test_ds)
    # true = dataframe['target'].values
    # print(pred)
    # print(true)


if decision_tree:
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType','LocationE', 'LocationN', 'Temperature', 'RainDur']
    df = dataframe[['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType', 'Temperature', 'RainDur']]
    X = df.drop(columns='RoadType')
    y = df['RoadType']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    dtc_y_predict = dtc.predict(X_test)
    accuracy_score(y_test, dtc_y_predict)

    confusion = pd.DataFrame(
        confusion_matrix(y_test, dtc_y_predict),
        columns=['Predicted Motorway', 'Predicted Expressway', 'Predicted Principal road','Predicted Minor road',
                 'Predicted Motorway side installation', 'Predicted Other'],
        index=['True Motorway', 'True Expressway', 'True Principal road', 'True Minor road',
               'True Motorway side installation', 'True Other']
    )
    pd.options.display.max_columns = 10
    pd.set_option('display.width', 200)
    print(confusion)

    fi_dtc = pd.DataFrame(dtc.feature_importances_,
                          index=list(X.columns),
                          columns=['importance'])
    fi_dtc_sorted = fi_dtc.sort_values('importance', ascending=False)
    print(fi_dtc_sorted)


if random_forest:
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType','LocationE', 'LocationN', 'Temperature', 'RainDur']
    df = dataframe[['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType', 'Temperature', 'RainDur']]
    X = df.drop(columns='RoadType')
    y = df['RoadType']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    rfc_y_predict = rfc.predict(X_test)
    accuracy_score(y_test, rfc_y_predict)

    confusion = pd.DataFrame(
        confusion_matrix(y_test, rfc_y_predict),
        columns=['Predicted Motorway', 'Predicted Expressway', 'Predicted Principal road', 'Predicted Minor road',
                 'Predicted Motorway side installation', 'Predicted Other'],
        index=['True Motorway', 'True Expressway', 'True Principal road', 'True Minor road',
               'True Motorway side installation', 'True Other']
    )
    print(confusion)

    fi_rfc = pd.DataFrame(rfc.feature_importances_,
                          index=list(X.columns),
                          columns=['importance'])
    fi_rfc_sorted = fi_rfc.sort_values('importance', ascending=False)
    print(fi_rfc_sorted)