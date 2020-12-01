import pathlib
import os
import json
import argparse

import numpy as np
import math
import librosa
from statistics import mode

import tensorflow.keras as keras


CURRENT_PATH = str(pathlib.Path(__file__).parent.absolute())
TARGET_FILE_PATH=CURRENT_PATH+"/test_data/max1min_croppedLatinAmerica.wav"
JSON_PATH = CURRENT_PATH + "/test_data/processed.json"
NN_MODEL_PATH = CURRENT_PATH + "/saved_models/NN_Music_Classification"
CNN_MODEL_PATH = CURRENT_PATH + "/saved_models/CNN_Music_Classification"
LSTM_MODEL_PATH = CURRENT_PATH + "/saved_models/LSTM_Music_Classification"

SAMPLE_RATE = 22050
DURATION = 30 # unit is seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    data={
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2 -> 2

     # load audio file
    signal, sr = librosa.load(TARGET_FILE_PATH, sr=SAMPLE_RATE)

    # process and store segment data (MFCC)
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s # s=0 -> 0
        finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment
        
        mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample],sr=SAMPLE_RATE, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc = mfcc.T

        # store mfcc for segment if it has the expected length
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            print("{}, segment:{}".format(TARGET_FILE_PATH, s+1))


    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)


'''Loads training dataset from json file.'''
def load_data(TRAINING_DATA_PATH):

    with open(TRAINING_DATA_PATH, "r") as TrainingData:
        data = json.load(TrainingData)

    # convert lists to numpy arrays <- mfcc dataset
    x = np.array(data["mfcc"])

    print("Load_Data successful")

    return  x


'''Predict a single sample using the trained model'''
def predict(model, x):

    x = x[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(x)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    if predicted_index == 0:
        predicted_label = "Asia"
    if predicted_index == 1:
        predicted_label = "LatinAmerica"
    if predicted_index == 2:
        predicted_label = "MiddleEastern"
    if predicted_index == 3:
        predicted_label = "Africa"

    print("Predicted label: {}".format(predicted_label))

    return predicted_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify models to run ...')
    parser.add_argument(
        '--NN', '-n',
        action='store_true',
        help="Use Simple Neural Network."
    )
    #parser.add_argument(
    #    '--CNN', '-c',
    #    action='store_true',
    #    help="Use Convonlutional Neural Network."
    #)
    parser.add_argument(
        '--LSTM', '-l',
        action='store_true',
        help="Use RNN-LSTM Neural Network."
    )
    parser.add_argument(
        '--ALL', '-a',
        action='store_true',
        help="Use all three models."
    )
    save_mfcc(TARGET_FILE_PATH, JSON_PATH, num_segments=10)
    x = load_data(JSON_PATH)
    args = parser.parse_args()
    if args.NN:
        loaded_NN_model = keras.models.load_model(NN_MODEL_PATH)
        predict(loaded_NN_model, x[1])
    #if args.CNN:
    #    loaded_CNN_model = keras.models.load_model(CNN_MODEL_PATH)
    #    predict(loaded_CNN_model, x[1])
    if args.LSTM:
        loaded_LSTM_model = keras.models.load_model(LSTM_MODEL_PATH)
        predict(loaded_LSTM_model, x[1])
    if args.ALL:
        predicted_label_list = []
        loaded_NN_model = keras.models.load_model(NN_MODEL_PATH)
        predicted_label_list.append(predict(loaded_NN_model, x[1]))
    #    loaded_CNN_model = keras.models.load_model(CNN_MODEL_PATH)
    #    predicted_label_list.append(predict(loaded_CNN_model, x[1]))
        loaded_LSTM_model = keras.models.load_model(LSTM_MODEL_PATH)
        predicted_label_list.append(predict(loaded_LSTM_model, x[1]))
        result = mode(predicted_label_list)
        print(result)
