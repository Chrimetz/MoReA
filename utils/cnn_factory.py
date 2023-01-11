import tensorflow as tf

import tensorflow_hub as hub
from tensorflow import keras
from NeuralNetworks import custome_cnn_generator

class cnn_factory:

    def get(self, name):
        if name == "resnet50":
            return self.__get_resnet50()
        elif name == "resnet101":
            return self.__get_resnet101()
        elif name == "resnet152":
            return self.__get_resnet152()
        elif name == "nasnetmobile":
            return self.__get_nasnetmobile()
        elif name == "nasnetlarge":
            return self.__get_nasnetlarge()
        elif name == "mobilenet":
            return self.__get_mobilenet()
        elif name == "inceptionv3":
            return self.__get_inceptionv3()
        elif name == "densenet201":
            return self.__get_densenet201()
        elif name == "densenet169":
            return self.__get_densenet169()
        elif name == "densenet121":
            return self.__get_densenet121()
        elif name == "efficientnetb0":
            return self.__get_efficientnetb0()
        elif name == "efficientnetb1":
            return self.__get_efficientnetb1()
        elif name == "efficientnetb2":
            return self.__get_efficientnetb2()
        elif name == "efficientnetb3":
            return self.__get_efficientnetb3()
        elif name == "efficientnetb4":
            return self.__get_efficientnetb4()
        elif name == "efficientnetb5":
            return self.__get_efficientnetb5()
        elif name == "efficientnetb6":
            return self.__get_efficientnetb6()
        elif name == "efficientnetb7":
            return self.__get_efficientnetb7()
        elif name == "vgg16":
            return self.__get_vgg16()
        elif name == "vgg19":
            return self.__get_vgg19()
        elif name == "resnet50v2":
            return self.__get_resnet50V2()
        elif name == "resnet101v2":
            return self.__get_resnet101V2()
        elif name == "resnet152v2":
            return self.__get_resnet152V2()
        elif name == "Xception":
            return self.__get_Xcetion()
        elif name == "InceptionResNetV2":
            return self.__get_InceptionResNetV2()
        elif name == "MobileNetV2":
            return self.__get_mobileNetV2()
        elif name == "m-r50x1":
            return self.__get_mr50x1()
        elif name == "m-r101x3":
            return self.__get_mr101x3()
        elif name == "m-r101x1":
            return self.__get_mr101x1()
        elif name == "m-r50x3":
            return self.__get_mr50x3()
        elif name == "m-r154x4":
            return self.__get_mr154x4()
        elif name == "mobilenetV3":
            return self.__get_mobileNetV3_small()
        elif name == "alexnet":
            return self.__get_AlexNet()
        elif name == "alexNetModify1":
            return self.__get_modify_AlexNet_1()
        elif name == "alexNetModify2":
            return self.__get_modify_AlexNet_2()
        
    def __get_mobileNetV3_small(self):
        m = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5")])
        m.build([None, 244,244,3])

        return m

    def __get_mr50x1(self):
        return hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/ilsvrc2012_classification/1", trainable=True)

    def __get_mr50x3(self):
        return hub.KerasLayer("https://tfhub.dev/google/bit/s-r50x3/ilsvrc2012_classification/1", trainable=True)

    def __get_mr101x1(self):
        return hub.KerasLayer("https://tfhub.dev/google/bit/s-r101x1/ilsvrc2012_classification/1", trainable=True)

    def __get_mr154x4(self):
        return hub.KerasLayer("https://tfhub.dev/google/bit/s-r152x4/ilsvrc2012_classification/1", trainable=True)

    def __get_mr101x3(self):
        return hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x3/ilsvrc2012_classification/1", trainable=True)

    def __get_resnet50(self):
        return tf.keras.applications.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    def __get_resnet50V2(self):
        return tf.keras.applications.ResNet50V2(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_resnet152(self):
        return tf.keras.applications.ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    def __get_resnet101(self):
        return tf.keras.applications.ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    def __get_resnet101V2(self):
        return tf.keras.applications.ResNet101V2(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_resnet152V2(self):
        return tf.keras.applications.ResNet152V2(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_nasnetmobile(self):
        return tf.keras.applications.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

    def __get_nasnetlarge(self):
        return tf.keras.applications.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

    def __get_mobilenet(self):
        return tf.keras.applications.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000, classifier_activation='softmax')

    def __get_inceptionv3(self):
        return tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation='softmax')

    def __get_densenet201(self):
        return tf.keras.applications.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    
    def __get_densenet169(self):
        return tf.keras.applications.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    def __get_densenet121(self):
        return tf.keras.applications.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    
    def __get_efficientnetb0(self):
        return tf.keras.applications.EfficientNetB0(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_efficientnetb1(self):
        return tf.keras.applications.EfficientNetB1(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_efficientnetb2(self):
        return tf.keras.applications.EfficientNetB2(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_efficientnetb3(self):
        return tf.keras.applications.EfficientNetB3(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_efficientnetb4(self):
        return tf.keras.applications.EfficientNetB4(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_efficientnetb5(self):
        return tf.keras.applications.EfficientNetB5(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
        
    def __get_efficientnetb6(self):
        return tf.keras.applications.EfficientNetB6(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_efficientnetb7(self):
        return tf.keras.applications.EfficientNetB7(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
        
    def __get_vgg16(self):
        return tf.keras.applications.VGG16(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_vgg19(self):
        return tf.keras.applications.VGG19(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_Xcetion(self):
        return tf.keras.applications.Xception(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
    
    def __get_InceptionResNetV2(self):
        return tf.keras.applications.InceptionResNetV2(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
    
    def __get_mobileNetV2(self):
        return tf.keras.applications.MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights="imagenet", input_tensor=None, pooling=None, classes=1000, classifier_activation="softmax")

    def __get_AlexNet(self):
        ccg = custome_cnn_generator.custome_cnn_generator()

        ccg.add_Layer([
            keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        return ccg.get_model()

    def __get_modify_AlexNet_1(self):
        ccg = custome_cnn_generator.custome_cnn_generator()

        ccg.add_Layer([
            keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        return ccg.get_model()
        
    def __get_modify_AlexNet_2(self):
        ccg = custome_cnn_generator.custome_cnn_generator()

        ccg.add_Layer([
            keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        return ccg.get_model()
    