import numpy as np
from tensorflow import keras

num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
#print("x_train shape:", x_train.shape)
#print(x_train.shape[0], "train samples")
#print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

index = 9

#print({"image": x_test[index]})
print(y_test[index])

import onnxruntime as rt
sess = rt.InferenceSession("models/mnist.onnx")
input_name = sess.get_inputs()[0].name
#print(input_name)
label_name = sess.get_outputs()[0].name
#print(label_name)

input_ = np.array([x_test[index]])

#print({input_name: input_})

#print(input_.shape)

pred_onx = sess.run([label_name], {input_name: input_})[0]

print(pred_onx)

print(keras.utils.to_categorical(np.argmax(pred_onx), 10))