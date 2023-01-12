import tensorflow as tf 
from tensorflow import keras
from tensorflow.python.keras.utils.version_utils import ModelVersionSelector

class custome_cnn_generator():

    MODEL = None

    def __init__(self):
        self.MODEL = keras.models.Sequential()

    def add_Layer(self, layers):
        for layer in layers:            
                self.MODEL.add(layer)
        
        self.MODEL.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

    def get_model(self):
        if self.MODEL:
            return self.MODEL