import keras
from keras import layers #import layers from keras
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras.models import Model


def build_mymodel(input_shape=(224, 224, 3)):
    #create a model based off the mnist model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(3, kernel_size=(7, 7), activation="relu"),  #downsize the image size
            layers.MaxPooling2D(pool_size=(2, 2)), 
            layers.Conv2D(10, kernel_size=(7, 7), activation="relu"), #downsize again
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(), #flatten into 1-D
            layers.Dropout(0.2), #dropout rate of 20% probability
            layers.Dense(1, activation="sigmoid"), #put it into the 1 element, and use sigmoid to turn it into 0 or 1
        ]
    )

    return model


def build_resnet50(input_shape=(224, 224, 3)):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling='avg',
        classifier_activation="sigmoid",
        )

    x = base_model.output
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    return model
