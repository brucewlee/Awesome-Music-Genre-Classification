import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow.keras as keras


'''
Uses a Convolution Neural Network for music genre classification task
Result can be found in `readme_images`
'''

'''function map

    load_data 
    prepare_dataset
    build_CNN_model
    predict
    plotter

'''

# choose style sheet for MatPlotLib
plt.style.use('ggplot')
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
color_list = [CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)


# path to json file (our training data) that stores MFCCs and genre labels from LAMA : World Musics Genre DataSet
CURRENT_PATH = str(pathlib.Path(__file__).parent.absolute())
TRAINING_DATA_PATH = CURRENT_PATH + "/LAMA_DataSet.json"
TRAINED_MODEL_SAVE_PATH = CURRENT_PATH + '/saved_models/CNN_Music_Classification'


'''Loads training dataset from json file.'''
def load_data(TRAINING_DATA_PATH):

    with open(TRAINING_DATA_PATH, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y


'''Loads data and splits it into train, validation and test sets.'''
def prepare_datasets(test_size, validation_size):

    # load data
    x, y = load_data(TRAINING_DATA_PATH)

    # create train, validation and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size, random_state=1)

    # add an axis to input sets
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test


'''Generates CNN model'''
def build_CNN_model(input_shape):

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 1st dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))),
    model.add(keras.layers.Dropout(0.3)),

    # 2nd dense layer
    model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


'''Predict a single sample using the trained model'''
def predict(model, x, y):

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    x = x[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(x)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


'''Plots accuracy/loss'''
def plotter(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["acc"], label="train accuracy")
    axs[0].plot(history.history["val_acc"], label="test accuracy")
    axs[0].set_ylabel("accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("error")
    axs[1].set_xlabel("epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("error eval")

    plt.tight_layout()
    plt.savefig(CURRENT_PATH + '/readme_images' + '/CNN_Music_Classification' + '.png', dpi=500)

    plt.show()


def save_model(model):
    model.save(TRAINED_MODEL_SAVE_PATH,save_format='h5')
    print("successfully saved")


if __name__ == "__main__":

    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_CNN_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    model.summary()

    # train model
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=200)

    # save model
    save_model(model)

    # plot accuracy/error for training and validation
    plotter(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # pick a sample to predict from the test set
    X_to_predict = x_test[1]
    y_to_predict = y_test[1]

    # predict sample
    predict(model, X_to_predict, y_to_predict)