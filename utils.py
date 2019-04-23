from scikits.talkbox.features import mfcc
import numpy as np
import scipy.io.wavfile as wav
from hmmlearn import hmm

#from python_speech_features import mfcc
#import librosa


class Word:
    def __init__(self, category, filename):
        self.category = category
        self.filename = filename
        self.mfcc_matrix = None # feature vector
        self.sample_rate, self.signal = wav.read(self.filename)

    def get_category(self):
        return self.category

    def set_category(self, category):
        self.category = category

    def get_filename(self):
        return self.filename

    def set_filename(self, filename):
        self.filename = filename

    def get_mfcc_matrix(self):
        return self.mfcc_matrix

    def set_mfcc_matrix(self):

        if self.filename is None:
            #throw error
            return


        #self.mfcc_matrix = librosa.feature.mfcc(y=self.sig.astype('float'), sr=self.rate, n_mfcc=13)
        #self.mfcc_matrix = mfcc(signal=self.sig, samplerate=self.rate, numcep=13, nfft=2048)

        self.mfcc_matrix = mfcc(self.signal, nwin=int(self.sample_rate * 0.03), fs=self.sample_rate, nceps=13)[0]
        self.mfcc_matrix = self.mfcc_matrix[~np.isnan(self.mfcc_matrix).any(axis=1)]
class WordHMM:
    def __init__(self, category):
        self.category = category
        self.training_data = list()
        self.wordhmm = None

        ####NEED TO UNDERSTAND THESE
        self.nComp = 5  # number of states
        self.nMix = 2  # number of mixtures
        self.covarianceType = 'diag'  # covariance type
        self.n_iter = 10  # number of iterations
        self.startprobPrior = None
        self.transmatPrior = None
        self.bakisLevel = 2

    def get_start_prob_prior(self):
        return self.startprobPrior

    def get_category(self):
        return self.category

    def get_training_data(self):
        return self.training_data

    def add_to_training_data(self, training_data):
        self.training_data.append(training_data)

    def initModelParam(self, nComp, nMix, covarianceType, n_iter, bakisLevel):
        ''' init params for hmm model '''

        self.nComp = nComp  # number of states
        self.nMix = nMix  # number of mixtures
        self.covarianceType = covarianceType  # covariance type
        self.n_iter = n_iter  # number of iterations
        self.bakisLevel = bakisLevel

        startprobPrior, transmatPrior = self.initByBakis(nComp, bakisLevel)

        self.startprobPrior = startprobPrior
        self.transmatPrior = transmatPrior

    def initByBakis(self, nComp, bakisLevel):
        ''' init start_prob and transmat_prob by Bakis model '''
        startprobPrior = np.zeros(nComp)
        startprobPrior[0: bakisLevel - 1] = 1. / (bakisLevel - 1)

        transmatPrior = self.getTransmatPrior(nComp, bakisLevel)

        return startprobPrior, transmatPrior



    def getTransmatPrior(self, nComp, bakisLevel):
        ''' get transmat prior '''
        transmatPrior = (1. / bakisLevel) * np.eye(nComp)

        for i in range(nComp - (bakisLevel - 1)):
            for j in range(bakisLevel - 1):
                transmatPrior[i, i + j + 1] = 1. / bakisLevel

        for i in range(nComp - bakisLevel + 1, nComp):
            for j in range(nComp - i - j):
                transmatPrior[i, i + j] = 1. / (nComp - i)

        return transmatPrior

    def getHmmModel(self):
        ''' get hmm model from training data '''

        # GaussianHMM
        #         model = hmm.GaussianHMM(numStates, "diag") # initialize hmm model

        # Gaussian Mixture HMM
        #model = hmm.GMMHMM(n_components=self.nComp, n_mix=self.nMix,
        #                   transmat_prior=self.transmatPrior, startprob_prior=self.startprobPrior,
        #                   covariance_type=self.covarianceType, n_iter=self.n_iter)
        model = hmm.GMMHMM(n_components=self.nComp, n_mix=self.nMix,
                           covariance_type=self.covarianceType, n_iter=self.n_iter)
        train = self.get_training_data()

        """

               def fit(self, X, lengths=None):

               Parameters
               ----------
               X : array-like, shape (n_samples, n_features)
                   Feature matrix of individual samples.

               lengths : array-like of integers, shape (n_sequences, )
                   Lengths of the individual sequences in ``X``. The sum of
                   these should be ``n_samples``.

               """

        lengths = list()
        data_combined = np.concatenate((train[0], train[1], train[2], train[3], train[4], train[5], train[6], train[7], train[8], train[9]))
        for example in train:
            lengths.append(example.shape[0])

        model.fit(data_combined, lengths)
        #####################
        print('###### Start Prob ######')
        print(model.startprob_)
        print('###### Transmat Prob ######')
        print(model.transmat_)
        print('###### Transmat Prior Prob ######')
        print(model.transmat_prior)
        self.wordhmm = model

        #model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))
