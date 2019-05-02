# !/usr/bin/env python

from prettytable import PrettyTable
from utils import Word, WordHMM
import os

#audio stuff
import pyaudio
import wave
from sys import byteorder
from array import array
from struct import pack
import copy


######################################
# Data loading functions
######################################


def load_data(data, category):
    os.chdir(category)
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".wav"):
            W = Word(category, filename)
            W.set_mfcc_matrix()
            data.append(W)
    os.chdir("..")


def load_all_training_data(training_data_list, category_labels):
    os.chdir("..//Training_Data")
    for category in category_labels:
        load_data(training_data_list, category)


def load_all_testing_data(testing_data_list, category_labels):
    os.chdir("..//Testing_Data")
    for category in category_labels:
        load_data(testing_data_list, category)

######################################
# HMM FUNCTIONS
######################################


def train_hmms(words, category_list):
    'Train the Hmm'
    word_hmms = list()

    # create an HMM for each category
    for category in category_list:
        w = WordHMM(category)
        word_hmms.append(w)

    for word_hmm in word_hmms:
        for training_word in words:
            if training_word.get_category() == word_hmm.get_category():
                word_hmm.add_to_training_data(training_word.get_mfcc_matrix())

        # get hmm model
        num_components = 5
        if word_hmm.get_category() == 'Play' or word_hmm.get_category() == 'Wayne':
            num_components = 3

        if word_hmm.get_category() == 'Drake':
            num_components = 4

        if word_hmm.get_category() == 'Kendrick' or word_hmm.get_category() == 'Gambino'\
                or word_hmm.get_category() == 'KendrickLamar':
            num_components = 6

        word_hmm.init_model_param(n_hidden_states=num_components, n_mixtures=3,
                                    covariance_type='diag', n_iter=10)
        word_hmm.get_hmm_model()

        normalize_categories(word_hmm)

    return word_hmms


def init_categories():
    artist_categories = list()
    artist_categories.append("Play")
    artist_categories.append("Drake")
    artist_categories.append("Kendrick")
    artist_categories.append("Chance")
    artist_categories.append("Kanye")
    artist_categories.append("Wayne")
    artist_categories.append("Snoop")
    artist_categories.append("Gambino")
    artist_categories.append("Eminem")
    artist_categories.append("MacMiller")
    artist_categories.append("PostMalone")

    # below categories added to improve performance
    artist_categories.append("LilWayne")
    artist_categories.append("KendrickLamar")

    return artist_categories


def predict(test_words, word_hmms):
    ''' recognition '''
    predicted_category_list = list()

    for artist in test_words:
        scores = list()

        for recognizer in word_hmms:
            score = recognizer.wordhmm.score(artist.get_mfcc_matrix())
            scores.append(score)

        idx = scores.index(max(scores))
        predicted_category = word_hmms[idx].get_category()
        predicted_category_list.append(predicted_category)

    return predicted_category_list


def get_classification_rate(actual_value_list, predicted_value_list):
    num_correct = 0
    length1 = len(actual_value_list)
    length2 = len(predicted_value_list)

    if length1 != length2:
        raise ValueError("Lengths of list are not equal")

    for i in range(length1):
        if actual_value_list[i] == predicted_value_list[i]:
            num_correct += 1

    return float(num_correct)/length1


def normalize_categories(artist):
    # normalize categories for words that had multiple training labels
    # Lil Wayne -> Wayne
    # Kendrick Lamar -> Kendrick
    # used to normalize both the HMM_category and the training / testing data categories

    if artist.get_category() == 'LilWayne':
        artist.set_category('Wayne')
    if artist.get_category() == 'KendrickLamar':
        artist.set_category('Kendrick')


######################################
# AUDIO RECORD FUNCTIONS
# https://stackoverflow.com/questions/892199/detect-record-audio-in-python
######################################
THRESHOLD = 500  # audio levels not normalised.
CHUNK_SIZE = 1024
SILENT_CHUNKS = 70
# SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
RATE = 44100
CHANNELS = 1
TRIM_APPEND = RATE / 4


def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD


def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r


def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return copy.deepcopy(data_all[_from:(_to + 1)])


def record():
    """Record a word or words from the microphone and
    return the data as an array of signed shorts."""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    silent_chunks = 0
    audio_started = False
    data_all = array('h')

    while True:
        # little endian, signed short
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        data_all.extend(data_chunk)

        silent = is_silent(data_chunk)

        if audio_started:
            if silent:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif not silent:
            audio_started = True

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    data_all = trim(data_all)  # we trim before normalize as threshhold applies to un-normalized wave (as well as is_silent() function)
    data_all = normalize(data_all)
    return sample_width, data_all


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wave_file = wave.open(path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(sample_width)
    wave_file.setframerate(RATE)
    wave_file.writeframes(data)
    wave_file.close()


if __name__ == '__main__':
    # LOAD TRAIN DATA
    print("Loading training data")
    categories = init_categories()
    training_data = list()
    load_all_training_data(training_data, categories)
    print("Done loading training data")

    # TRAIN MODELS
    print("Training the word HMMs")
    hip_hop_hmms = train_hmms(training_data, categories)
    print("Done training the word HMMs")

    # LOAD TESTING DATA
    print('Loading Testing Data')
    testing_data = list()
    load_all_testing_data(testing_data, categories)
    print('Done Loading Testing Data')

    # Create true category labels for testing data
    true_category_list = list()
    for word in testing_data:
        # normalize categories for words that had multiple training
        normalize_categories(word)
        true_category_list.append(word.get_category())

    # PREDICT TESTING DATA
    print("Predicting Testing Data")
    prediction = predict(testing_data, hip_hop_hmms)
    print("Done Predicting Testing Data")

    # present data so easy to see which examples are being marked incorrectly
    t = PrettyTable(['Example Number', 'Real Value', 'Predicted Value'])
    misclassified = PrettyTable(['Example Number', 'Real Value', 'Predicted Value'])

    for i in range(len(testing_data)):
        t.add_row([i+1, testing_data[i].get_category(), prediction[i]])
        if testing_data[i].get_category() != prediction[i]:
            misclassified.add_row([i+1, testing_data[i].get_category(), prediction[i]])

    print("Overall classification rate is {}".format(get_classification_rate(prediction, true_category_list)))
    print(t)
    print(misclassified)
"""
demoList = list()
for i in range(10):
    print("please speak a word into the microphone")
    filename = 'demo' + str(i) + '.wav'
    record_to_file(filename)
    print("done - result written to ", filename)
    demo = Word('Test', filename)
    demo.set_mfcc_matrix()
    demoList.append(demo)
    demoList2 = list()
    demoList2.append(demo)
    predicted_artist = predict(demoList2, hip_hop_hmms)
    print("Predicted Artist", predicted_artist[0])

predictedWords = predict(demoList, hip_hop_hmms)
for i in range(10):
    print("Predicted Word is: ", predictedWords[i])
"""