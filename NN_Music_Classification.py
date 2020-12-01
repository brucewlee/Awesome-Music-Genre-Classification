import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow.keras as keras


'''
Uses simple Keras NN for music genre classification task
Some results can be found in `readme_images`
'''

'''function map

    load_data 
    build_NN_model
    predict
    plotter
    save_model

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
TRAINED_MODEL_SAVE_PATH = CURRENT_PATH + '/saved_models/NN_Music_Classification'


'''Loads training dataset from json file.'''
def load_data(TRAINING_DATA_PATH):

    with open(TRAINING_DATA_PATH, "r") as TrainingData:
        data = json.load(TrainingData)

    # convert lists to numpy arrays <- mfcc dataset
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Load_Data successful")

    return  x, y


'''Generates NN model'''
def build_NN_model(input_shape):
    # build neural network
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=input_shape),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


'''Predict a single sample using the trained model'''
def predict(model, x, y):

    x = x[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(x)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


def plotter(history):
    fig, axes = plt.subplots(2)

    # accuracy plot
    axes[0].plot(history.history["acc"], label = "train accuracy")
    axes[0].plot(history.history["val_acc"], label = "test accuracy")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="lower right")
    axes[0].set_title("accuracy eval")

    # error plot
    axes[1].plot(history.history["loss"], label = "train error")
    axes[1].plot(history.history["val_loss"], label = "test error")
    axes[1].set_ylabel("error")
    axes[1].set_xlabel("epoch")
    axes[1].legend(loc="upper right")
    axes[1].set_title("error eval")
    plt.tight_layout()
    plt.savefig(CURRENT_PATH + '/readme_images' + '/Simple_NN_Music_Classification' + '.png', dpi=500)

    plt.show()


def save_model(model):
    model.save(TRAINED_MODEL_SAVE_PATH,save_format='h5')
    print("successfully saved")


if __name__ == "__main__":

    # load data
    x, y = load_data(TRAINING_DATA_PATH)

    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    
    # create network
    input_shape = (x.shape[1], x.shape[2])
    model = build_NN_model(input_shape)

    # compile 
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    # give summary
    model.summary()

    # train model
    progression = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=200)

    # save model
    save_model(model)

    # plot training progression
    plotter(progression)

    # pick a sample to predict from the test set
    X_to_predict = x_test[1]
    y_to_predict = y_test[1]

    # predict sample
    predict(model, X_to_predict, y_to_predict)