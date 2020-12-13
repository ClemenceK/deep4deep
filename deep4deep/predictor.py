from deep4deep.w2v_embedding_and_rnn_model import Preprocessor, Embedder, pad_X

from tensorflow import keras
from os import path

class Predictor():

    def __init__(self, preprocessor, embedder, path_to_model="deeptech_NLP_model"):
        print("initializing Predictor")
        self.model = keras.models.load_model(path_to_model)
        self.preprocessor = preprocessor
        self.embedder = embedder

    def predict(self, a_series_of_text_snippets):
        '''
        receives a Panda Series of text presenting companies (string)
        returns the trained model's prediction as to
        whether the company thus described is a deeptech or not
        '''
        X = self.preprocessor.transform(a_series_of_text_snippets)
        X = self.embedder.transform(X)
        X = pad_X(X, dtype='float32')
        y_pred = self.model.predict(X)
        return y_pred
