from scikits.talkbox.features import mfcc
import numpy as np
import scipy.io.wavfile as wav
from hmmlearn import hmm

class Word:
    def __init__(self, category, filename):
        self.category = category
        self.filename = filename
        self.mfcc_matrix = None  # feature vector
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
        self.mfcc_matrix = mfcc(self.signal, nwin=int(self.sample_rate * 0.03), fs=self.sample_rate, nceps=13)[0]
        self.mfcc_matrix = self.mfcc_matrix[~np.isnan(self.mfcc_matrix).any(axis=1)]


class WordHMM:
    def __init__(self, category):
        self.category = category
        self.training_data = list()
        self.wordhmm = None

        self.n_hidden_states = 5  # number of states
        self.n_mixtures = 3  # number of mixtures
        self.covariance_type = 'diag'  # covariance type
        self.n_iter = 10  # number of iterations

    def get_category(self):
        return self.category

    def set_category(self, category):
        self.category = category

    def get_training_data(self):
        return self.training_data

    def add_to_training_data(self, training_data):
        self.training_data.append(training_data)

    def init_model_param(self, n_hidden_states, n_mixtures, covariance_type, n_iter):
        """initialize model parameters for hmm model """

        self.n_hidden_states = n_hidden_states  # number of states
        self.n_mixtures = n_mixtures  # number of mixtures
        self.covariance_type = covariance_type  # covariance type
        self.n_iter = n_iter  # number of iterations

    def get_hmm_model(self):
        """ get hmm model from training data """

        # Gaussian Mixture HMM
        model = hmm.GMMHMM(n_components=self.n_hidden_states, n_mix=self.n_mixtures,
                           covariance_type=self.covariance_type, n_iter=self.n_iter)
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
        self.wordhmm = model