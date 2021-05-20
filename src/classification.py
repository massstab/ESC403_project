# -*- coding: utf-8 -*-
"""
Created on 13 May 2021
@author: dtm
"""
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from datasets import data_all as dataframe
from helpers import df_to_dataset

dataframe = dataframe.drop(columns=['Date', 'SumBikerNumber',
                                    'SumBikerNumber', 'SumCars', 'SumPedestrianNumber'])
# Shorten feature names just for convenient output format
new_cols = {"AccidentSeverityCategory": "Severity", "AccidentType": "AccType", "AvgTemperature": "Temperature", "AvgRainDur": "RainDur",
            "AccidentInvolvingPedestrian": "Pedestrian", "AccidentInvolvingBicycle": "Bicycle",
            "AccidentInvolvingMotorcycle": "Motorcycle", "AccidentLocation_CHLV95_E": "LocationE",
            "AccidentLocation_CHLV95_N": "LocationN"}
dataframe.rename(columns=new_cols, inplace=True)
dataframe.loc[dataframe['RoadType'] == 9, 'RoadType'] = 5
print(dataframe['RoadType'])

dataframe['target'] = dataframe['Bicycle'].values

# Split into train test and validation
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

feature_columns = []
# Define the feature columns
# for header in ['AccType', Severity', 'Pedestrian', 'Bicycle', 'Motorcycle', 'RoadType', 'LocationE', 'LocationN', 'Temperature', 'RainDur']:
for header in ['Pedestrian', 'RoadType', 'Motorcycle', 'AccType', 'Severity']:
    feature_columns.append(feature_column.numeric_column(header))

# Use DenseFeatures layer as an input to the Keras model
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


batch_size = 2048
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# for feature_batch, label_batch in train_ds.take(1):
#     print('Every feature:', list(feature_batch.keys()))
#     print('A batch of RoadType:', feature_batch['RoadType'])
#     print('A batch of Bicycle:', feature_batch['Bicycle'])
#     print('A batch of Motorcycle:', feature_batch['Motorcycle'])
#     print('A batch of targets:', label_batch)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.2),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=None)
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
