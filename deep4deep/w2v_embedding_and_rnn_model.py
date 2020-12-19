import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import gensim.downloader as api


#from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import TransformerMixin, BaseEstimator


#from deep4deep.text_retrieval import prepare_my_df, get_meta_description_columns
from deep4deep.text_processing import text_preprocessing
#from deep4deep.utils import simple_time_tracker

MASK_VALUE = -100
EMBEDDING = "glove-wiki-gigaword-300"
# glove-wiki-gigaword-300
# also tried glove-twitter-200, not better


def pad_X(X, dtype='float32'):
    X_pad = pad_sequences(X,
                  dtype=dtype,
                  padding='post',
                  value=MASK_VALUE)
    return X_pad

def init_and_compile_model():
    # –– Initialization
    model = models.Sequential()
    model.add(layers.Masking(mask_value=MASK_VALUE))

    model.add(layers.LSTM(units=12, activation='tanh', return_sequences=True)) # NEED True if and only if you add another LSTM after
    # dropout1
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.LSTM(units=20, activation='tanh', return_sequences=False)) # need FALSE for last LSTM layer
    # dropout2
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(10, activation='relu'))
    # dropout3
    #model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation='linear'))

    # –– Compilation
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse', 'accuracy'])
    return model



class Preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        print("initializing Preprocessor")

    def fit(self, X, y=None):
        #no fit as using the wiki-gigaword embedding
        return self

    def transform(self, X, y=None):
        X_new = X.map(text_preprocessing)
        return X_new


class Embedder(TransformerMixin, BaseEstimator):

    def __init__(self):
        """define parameters useful for fit and transform"""
        print("initializing Embedder")
        self.history = None
        print(f"downloading transfer learning embedder {EMBEDDING} – this may take time… coffee break? ☕️☕️☕️")
        self.embedding = api.load(EMBEDDING)
        # Downloads the corpus, unless it is already cached on your local machine
        # then loads it (can still take a few minutes)


    def embed_text(self, tokenized_text):
        result = []
        for word in tokenized_text:
            if word in self.embedding.wv:
                vector = self.embedding.wv.get_vector(word)
                result.append(vector)
        if len(result) == 0:
            result = [np.ones(self.embedding.vector_size)*MASK_VALUE]
        return result

    def fit(self, X, y=None):
        #no fit as using the wiki-gigaword embedding
        return self

    def transform(self, X, y=None):
        X_new = X.map(self.embed_text)
        # X.iloc[:,0] to make X a series
        return X_new



class LstmModel(TransformerMixin, BaseEstimator):

    def __init__(self):
        # size of vocabulary, not needed now
        # self.vocab_size = len(self.embedding.wv.vocab)
        # size of embedding vectors, not needed now
        # self.size_embedding = self.embedding.vector_size
        # LSTM model
        print("initializing LstmModel")
        self.model = init_and_compile_model()
        self.es = EarlyStopping(patience=15, restore_best_weights=True)

    def fit(self, X, y, X_val, y_val):
        X_pad = pad_X(X)
        X_val =  pad_X(X_val)
        self.history = self.model.fit(X_pad, y, epochs= 500, batch_size=64, callbacks=[self.es], validation_data=(X_val, y_val))
        return self

    def predict(self, X):
        # need to prepocess and embed X (done in parent's predictor)
        X_pad = pad_X(X, dtype='float32')
        y = self.model.predict(X_pad)
        return y


#if __name__ == "__main__":

    # initial df
    #my_df = prepare_my_df(df)
    #my_df = get_meta_description_columns(my_df)

    # Importing meta-descriptions from websites and preprocessing
