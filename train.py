import tensorflow as tf
import numpy as np
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc

import model 

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

path_mean = np.array([94.07080238, 82.69394267, 98.84364401])
path_std = np.array([23.95747985, 29.23472607, 20.76279442])


def normalization(image):
    path_mean = np.array([94.07080238, 82.69394267, 98.84364401])
    path_std = np.array([23.95747985, 29.23472607, 20.76279442])
    return (image - path_mean) / (path_std + 1e-7)

def denormalize(image, mean, std):
    mean = tf.reshape(tf.constant(mean), (1, 1, 1, 3))
    std = tf.reshape(tf.constant(std), (1, 1, 1, 3))
    return image * std + mean

def custom_preprocessing(image, path_mean, path_std, augmentation=False):
    if augmentation:
        image = tf.image.random_brightness(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2) 
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.1) 

    image = (image - path_mean) / (path_std + 1e-7)
    return image

def get_model(model_name):
    try:
        # Dynamically get the corresponding model function from model.py
        model_function = getattr(model, f'build_{model_name}')
        return model_function(input_shape=(224, 224, 3))
    except AttributeError as e:
        raise ValueError(f"Model {model_name} not found. Error: {str(e)}")


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name to train.')
    parser.add_argument('-e', '--epoch', type=int, default=50, help='Number of epoches to train.')
    args = parser.parse_args()

    # Get the specified model
    model = get_model(args.model)

    data_dir = './data/MHIST'

    train_datagen = ImageDataGenerator(
        rotation_range=20,  
        width_shift_range=0.05,  
        height_shift_range=0.05,  
        shear_range=0.05,  
        zoom_range=0.05,  
        horizontal_flip=True,  
        fill_mode='nearest',
        preprocessing_function=normalization
    )

    valid_datagen = ImageDataGenerator(
        # rescale=1./255,  
        preprocessing_function=normalization
    )

    annotations = pd.read_csv(os.path.join(data_dir, "annotations.csv")) #import the spreadsheet and save it into a variable

    train_df = annotations[annotations['Partition'] == 'train']
    test_df = annotations[annotations['Partition'] == 'test']

    train_loader = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='./data/MHIST/images',
        x_col="Image Name",
        y_col="Majority Vote Label",
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    test_loader = valid_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='./data/MHIST/images',
        x_col="Image Name",
        y_col="Majority Vote Label",
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    batch_size = 128

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        x=train_loader,
        y=None,
        batch_size=batch_size,
        epochs=args.epoch,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=test_loader,
        shuffle=True,
        class_weight={0:0.8, 1:1.2},
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    )

    score = model.evaluate(test_loader, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    testpredictions = []
    testgroundtrues = []

    for x,y in test_loader:
        y_pred = model(x)
        testpredictions.append(y_pred)
        testgroundtrues.append(y)

    testpredictions = np.concatenate(testpredictions)
    testgroundtrues = np.concatenate(testgroundtrues)

    print("The accuracy score for the test loader is: ", accuracy_score(testgroundtrues, testpredictions.round()))
    print("The F1 score for the test loader is: ", f1_score(testgroundtrues, testpredictions.round()), "\n")

    conf = confusion_matrix(testgroundtrues, testpredictions.round())
    print("The confusion matrix for the test loader is: \n", conf, "\n")
    print("The sensitivity score of SSA for the test loader is: ", conf[0,0]/(conf[0,0]+conf[0,1]))
    print("The specificity score of HP for the test loader is: ", conf[1,1]/(conf[1,1]+conf[1,0]))

    #calculate ROC curve for test loader

    fpr, tpr, thresholds = roc_curve(testgroundtrues, testpredictions) 
    roc_auc = auc(fpr, tpr)
    print("The AUC score of the test loader is: ", roc_auc)

    # plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='Train ROC curve' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Training')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Save the accuracy plot as a file
    plt.savefig('ROC.png')
    plt.clf()  # Clear the current figure to avoid overlap


    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the accuracy plot as a file
    plt.savefig('model_accuracy.png')
    plt.clf()  # Clear the current figure to avoid overlap


    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the loss plot as a file
    plt.savefig('model_loss.png')
    plt.clf()  # Clear the current figure

    print("DONE!!")

if __name__ == '__main__':
    main()