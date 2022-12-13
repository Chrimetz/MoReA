import onnx
import tf2onnx

# Importing the Boston Housing dataset
from sklearn.datasets import load_boston

import tensorflow as tf

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

# Loading the Boston Housing dataset
boston = load_boston()

# Initializing the dataframe
data = pd.DataFrame(boston.data)

#Adding the feature names to the dataframe
data.columns = boston.feature_names

#Adding target variable to dataframe
data['PRICE'] = boston.target
data.head()

# Split the data into train and test with 80 train / 20 test
train,test = train_test_split(data, test_size=0.2, random_state = 1)
train,val = train_test_split(train, test_size=0.2, random_state = 1)

# Helper functions
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
def format_output(data):
    y1 = data.pop('PRICE')
    y1 = np.array(y1)
    y2 = data.pop('PTRATIO')
    y2 = np.array(y2)
    return y1, y2

# Get PRICE and PTRATIO as the 2 outputs and format them as np 
# arrays
# PTRATIO - pupil-teacher ratio by town
train_stats = train.describe()
train_stats.pop('PRICE')
train_stats.pop('PTRATIO')
train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)
val_Y = format_output(val)

# Normalize the training and test data
norm_train_X = np.array(norm(train))
norm_test_X = np.array(norm(test))
norm_val_X = np.array(norm(val))

def build_model():
    # Define model layers.
    input_layer = tf.keras.layers.Input(shape=(len(train .columns),))
    first_dense = tf.keras.layers.Dense(units='128', activation='relu')(input_layer)
    # Y1 output will be fed from the first dense
    y1_output = tf.keras.layers.Dense(units='1', name='price_output')(first_dense)
    second_dense = tf.keras.layers.Dense(units='128',activation='relu')(first_dense)
    # Y2 output will be fed from the second dense
    y2_output = tf.keras.layers.Dense(units='1',name='ptratio_output')(second_dense)
    # Define the model with the input layer 
    # and a list of output layers
    model = tf.keras.Model(inputs=input_layer,outputs=[y1_output, y2_output])

    return model

model = build_model()
# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.keras.optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer,
              loss={'price_output': 'mse', 'ptratio_output': 'mse'},
 metrics={'price_output':tf.keras.metrics.RootMeanSquaredError(),
          'ptratio_output':tf.keras.metrics.RootMeanSquaredError()})

# Train the model for 100 epochs
history = model.fit(norm_train_X, train_Y,
   epochs=100, batch_size=10, validation_data=(norm_test_X, test_Y))

# Test the model and print loss and rmse for both outputs
loss,Y1_loss,Y2_loss,Y1_rmse,Y2_rmse=model.evaluate(x=norm_val_X, y=val_Y)
print()
print(f'loss: {loss}')
print(f'price_loss: {Y1_loss}')
print(f'ptratio_loss: {Y2_loss}')
print(f'price_rmse: {Y1_rmse}')
print(f'ptratio_rmse: {Y2_rmse}')

boston_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(boston_model, "boston.onnx")