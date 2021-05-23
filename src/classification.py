# -*- coding: utf-8 -*-
"""
Created on 13 May 2021
@author: dtm
according to this tutorial: https://www.tensorflow.org/tutorials/structured_data/feature_columns
"""

import logging, os
import matplotlib.pyplot as plt
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
from sklearn.tree import plot_tree


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
decision_tree = False
random_forest = True


if sequential_model:
    print('starting sequential model classification')
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType', 'LocationE', 'LocationN', 'Temperature', 'RainDur']
    features = ['AccType', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType']
    target = 'Severity'
    if target == 'Severity':
        dataframe['Severity'] = dataframe['Severity'] - 1
    df = dataframe[features+[target]].copy()
    df.rename(columns={target: 'target'}, inplace=True)

    # Split into train test and validation
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    batch_size = 128
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    feature_columns = []
    # Define the feature columns
    for header in features:
        feature_columns.append(feature_column.numeric_column(header))

    print(feature_columns)
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
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.3),
        layers.Dense(4, activation='softmax')
    ])

    optimizer = optimizers.Adam(learning_rate=0.01)  # Defines the learning rate
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

    history = model.fit(train_ds, validation_data=val_ds, epochs=60, callbacks=None)
    loss, accuracy = model.evaluate(test_ds)
    # print("Accuracy", accuracy)
    # print(dataframe.info())
    # pred = model.predict_classes(test_ds)
    # true = dataframe['target'].values
    # print(pred)
    # print(true)


if decision_tree:
    print('starting decision tree classification')
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
    print('starting random forest classification')
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType','LocationE', 'LocationN', 'Temperature', 'RainDur']
    features = ['AccType', 'RoadType', 'Pedestrian', 'Bicycle', 'Motorcycle', 'Temperature', 'RainDur']
    # class_names = ['Motorway', 'Expressway', 'Principal road', 'Minor road', 'Motorway side installation', 'Other']
    class_names = ['Accident with fatalities', 'Accident with severe injuries',
                   'Accident with light injuries', 'Accident with property damage']
    X = dataframe[features]
    y = dataframe['Severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    rfc = RandomForestClassifier(random_state=1, max_depth=3)
    rfc.fit(X_train, y_train)
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=3, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=100, n_jobs=None, oob_score=False,
                          random_state=None, verbose=0, warm_start=False)
    estimator = rfc.estimators_[0]

    from subprocess import call

    fig = plt.figure(figsize=(21, 4), dpi=144)
    plt.style.use('seaborn')
    plot_tree(estimator, fontsize=7, feature_names=features, class_names=class_names, filled=True)
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("../presentation/figures/forest_tree.png", bbox_inches='tight',
                pad_inches=0)

    rfc_y_predict = rfc.predict(X_test)
    print(accuracy_score(y_test, rfc_y_predict))

    confusion = pd.DataFrame(
        confusion_matrix(y_test, rfc_y_predict),
        columns=class_names,
        index=['True ' + i for i in class_names]
    )

    fi_rfc = pd.DataFrame(rfc.feature_importances_,
                          index=list(X.columns),
                          columns=['importance'])
    fi_rfc_sorted = fi_rfc.sort_values('importance', ascending=False)

    print(fi_rfc_sorted)
    print(confusion)
