# -*- coding: utf-8 -*-
"""
Created on 13 May 2021
@author: dtm
according to this tutorial: https://www.tensorflow.org/tutorials/structured_data/feature_columns and session_06b
"""

import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence tensorflow a bit
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split
from datasets import data_all as dataframe
from helpers import df_to_dataset, prepare_data_classification, save_tree

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# set some pandas option to use the whole terminal width to display results
pd.options.display.max_columns = 10
pd.set_option('display.width', 200)

dataframe = prepare_data_classification(dataframe)

sequential_model = True
decision_tree = True
random_forest = True

if sequential_model:
    print('----------starting sequential model classification----------')
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType', 'LocationE', 'LocationN', 'Temperature', 'RainDur']
    features = ['AccType', 'Severity', 'Pedestrian', 'Bicycle']
    target = 'RoadType'
    print('Features: ', features)
    print('Target: ', target)
    if target == 'Severity':
        dataframe['Severity'] = dataframe['Severity'] - 1
    df = dataframe[features + [target]].copy()
    df.rename(columns={target: 'target'}, inplace=True)

    # Split into train test and validation
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    batch_size = 1024
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    feature_columns = []
    # Define the feature columns
    for header in features:
        feature_columns.append(feature_column.numeric_column(header))

    # Use DenseFeatures layer as an input to the Keras model
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # Take a look at a batch of a feature
    # for feature_batch, label_batch in train_ds.take(1):
    #     print('Every feature:', list(feature_batch.keys()))
    #     print('A batch of RoadType:', feature_batch['RoadType'])
    #     print('A batch of Bicycle:', feature_batch['Bicycle'])
    #     print('A batch of Motorcycle:', feature_batch['Motorcycle'])
    #     print('A batch of targets:', label_batch)

    num_units = dataframe[target].nunique()
    print(num_units)
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.3),
        layers.Dense(num_units, activation='softmax')
    ])

    optimizer = optimizers.Adam(learning_rate=0.01)  # Defines the learning rate
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=None)
    loss, accuracy = model.evaluate(test_ds)

if decision_tree:
    print('\n----------starting decision tree classification----------')
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType','LocationE', 'LocationN', 'Temperature', 'RainDur']
    features = ['Severity', 'AccType', 'Pedestrian', 'Bicycle', 'Motorcycle', 'Temperature', 'RainDur']
    # TODO: Adapt class_names for different target values
    class_names = ['Accident with fatalities', 'Accident with severe injuries',
                   'Accident with light injuries', 'Accident with property damage']
    target = 'RoadType'
    X = dataframe[features]
    y = dataframe[target]
    print('Features: ', features)
    print('Target: ', target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    dtc = DecisionTreeClassifier(random_state=1)
    dtc.fit(X_train, y_train)

    dtc_y_predict = dtc.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, dtc_y_predict))

    # confusion = pd.DataFrame(
    #     confusion_matrix(y_test, dtc_y_predict),
    #     columns=class_names,
    #     index=['True ' + i for i in class_names]
    # )
    # print(confusion)
    fi_dtc = pd.DataFrame(dtc.feature_importances_,
                          index=list(X.columns),
                          columns=['importance'])
    fi_dtc_sorted = fi_dtc.sort_values('importance', ascending=False)
    print(fi_dtc_sorted)

if random_forest:
    print('\n----------starting random forest classification----------')
    # all features: ['AccType', 'Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType','LocationE', 'LocationN', 'Temperature', 'RainDur']
    features = ['Severity', 'AccType', 'Pedestrian', 'Bicycle', 'Motorcycle', 'Temperature', 'RainDur']
    # TODO: Adapt class_names for different target values
    class_names = ['Accident with fatalities', 'Accident with severe injuries',
                   'Accident with light injuries', 'Accident with property damage']
    target = 'RoadType'
    X = dataframe[features]
    y = dataframe['Severity']
    print('Features: ', features)
    print('Target: ', target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    rfc = RandomForestClassifier(random_state=1, max_depth=5)
    # rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
    #                              max_depth=None, max_features='auto', max_leaf_nodes=None,
    #                              max_samples=None, min_impurity_decrease=0.0,
    #                              min_impurity_split=None, min_samples_leaf=1,
    #                              min_samples_split=2, min_weight_fraction_leaf=0.0,
    #                              n_estimators=100, n_jobs=None, oob_score=False,
    #                              random_state=1, verbose=0, warm_start=False)
    rfc.fit(X_train, y_train)
    est = rfc.estimators_[3]
    save_tree(est, features, class_names)


    rfc_y_predict = rfc.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, rfc_y_predict))

    # confusion = pd.DataFrame(
    #     confusion_matrix(y_test, rfc_y_predict),
    #     columns=class_names,
    #     index=['True ' + i for i in class_names]
    # )
    # print(confusion)
    fi_rfc = pd.DataFrame(rfc.feature_importances_,
                          index=list(X.columns),
                          columns=['importance'])
    fi_rfc_sorted = fi_rfc.sort_values('importance', ascending=False)

    print(fi_rfc_sorted)
