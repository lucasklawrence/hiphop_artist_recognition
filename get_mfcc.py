# !/usr/bin/env python

#from python_speech_features import mfcc
#from python_speech_features import delta
#from python_speech_features import logfbank
import scipy.io.wavfile as wav
#import librosa
from utils import Word, WordHMM
import os

cwd = os.getcwd()
print(cwd)
#(rate, sig) = wav.read("..\\Drake\\Drake.wav")

def load_training_data(training_data, category):
    os.chdir("..")
    os.chdir(category)
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".wav"):
            W = Word(category, filename)
            W.set_mfcc_matrix()
            training_data.append(W)


def load_all_training_data(training_data):
    load_training_data(training_data, "Play")
    print("added play")
    load_training_data(training_data, "Drake")
    print("added drake")
    load_training_data(training_data, "Kendrick")
    load_training_data(training_data, "Chance")
    load_training_data(training_data, "Kanye")
    load_training_data(training_data, "Wayne")
    load_training_data(training_data, "Snoop")
    load_training_data(training_data, "Gambino")

def train_hmms(words, categories):
    'Train the Hmm'
    word_hmms = list()

    #create an HMM for each category
    for category in categories:
        W = WordHMM(category)
        word_hmms.append(W)

    for word_hmm in word_hmms:
        for word in words:
            if word.get_category() == word_hmm.get_category():
                word_hmm.add_to_training_data(word.get_mfcc_matrix())

        # get hmm model
        word_hmm.initModelParam(nComp=9, nMix=2, \
                                        covarianceType='diag', n_iter=10, \
                                        bakisLevel=2)
        word_hmm.getHmmModel()

    return word_hmms

def init_categories():
    categories = list();
    categories.append("Play")
    categories.append("Drake")
    categories.append("Kendrick")
    categories.append("Chance")
    categories.append("Kanye")
    categories.append("Wayne")
    categories.append("Snoop")
    categories.append("Gambino")
    #categories.append("Quiet")

    return categories


def recognize(test_words, word_hmms):
    ''' recognition '''
    predictCategoryIdList = []

    for testSpeech in test_words:
        scores = []

        for recognizer in word_hmms:
            score = recognizer.hmmModel.score(testSpeech.features)
            scores.append(score)

        idx = scores.index(max(scores))
        predictCategoryId = word_hmms[idx].get_category()
        predictCategoryIdList.append(predictCategoryId)

    return predictCategoryIdList

# LOAD TRAIN DATA
print("Loading training data")
categories = init_categories()
training_data = list()
load_all_training_data(training_data)
print("Done loading training data")

# TRAIN MODELS
print("Training the word HMMs")
hip_hop_hmms = train_hmms(training_data, categories)
print("Done training the word HMMs")

for hmm in hip_hop_hmms:
    print(hmm.category)
    print(hmm.startprobPrior)
    print(hmm.transmatPrior)

### Step.3 Loading test data
#print 'Step.3 Test data loading...',
#testDir = './test_data/'
#testSpeechList = loadData(testDir)
#print 'done!'
os.chdir("..")
os.chdir("Train")

testing_data = list()
chanceTest = Word("Chance", 'ChanceTest1.wav')
chanceTest.set_mfcc_matrix()
testing_data.append(chanceTest)
### Step.4 Recognition
print('Step.4 Recognizing...')
scores = list()
for recognizer in hip_hop_hmms:
    print(sum(recognizer.get_start_prob_prior()))
    score = recognizer.wordhmm.score(chanceTest.get_mfcc_matrix())
    scores.append(score)

idx = scores.index(max(scores))
predictCategoryId = hip_hop_hmms[idx].get_category()
print(predictCategoryId)


"""
os.chdir("..\\Drake")
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".wav"):
        W = Word("Drake", filename)
        W.set_mfcc_matrix()
        training_data.append(W)
        #(rate, sig) = wav.read(filename)
        #mfcc_feat=mfcc(sig, rate, nfft=2048)
        #d_mfcc_feat = delta(mfcc_feat, 2)
        #fbank_feat = logfbank(sig, rate, nfft=2048)
        #print(mfcc_feat.shape)
        #print(d_mfcc_feat.shape)

os.chdir("..\\Kendrick")
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".wav"):
        (rate, sig) = wav.read(filename)
        mfcc_feat=mfcc(sig, rate, nfft=2048)
        d_mfcc_feat = delta(mfcc_feat, 2)
        fbank_feat = logfbank(sig, rate, nfft=2048)
        print(mfcc_feat.shape)
        print(d_mfcc_feat.shape)

#mfcc_feat = mfcc(sig, rate, nfft=2048)
#d_mfcc_feat = delta(mfcc_feat, 2)
#fbank_feat = logfbank(sig, rate, nfft=2048)

print(fbank_feat[1:3, :])

print(mfcc_feat.shape)
"""