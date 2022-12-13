import onnx
import tf2onnx

# Importing the Boston Housing dataset
from sklearn.datasets import load_boston

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


# Compute the prediction with ONNX Runtime
import onnxruntime as rt
sess = rt.InferenceSession("boston.onnx")

print([x.name for x in sess.get_outputs()])

input_name = sess.get_inputs()[0].name
print(input_name)
label_name1 = sess.get_outputs()[0].name
print(label_name1)
label_name2 = sess.get_outputs()[1].name
print(label_name2)

input_feed = {input_name: [np.array([-0.39592582,0.85374287,-1.30031899,-0.29541676 ,-0.68858289 , 0.82781921
  ,0.07100048 ,-0.31650001, -0.30917602,-1.08348329, 0.42300873 ,-0.71757293], dtype=float)]}

pred_onx = sess.run(output_names=[o.name for o in sess.get_outputs()], input_feed=input_feed)

result = {}
for i in range(len(pred_onx)):
    result[sess.get_outputs()[i].name] = pred_onx[i].tolist()

print(result)