import pandas as pd
from sklearn.model_selection import train_test_split
from os import path
import joblib



from deep4deep.utils import simple_time_tracker
from deep4deep.text_retrieval import prepare_my_df, get_meta_description_columns,get_meta_description,get_dealroom_meta_description
from deep4deep.description_dataframe_preparation import data_prep, change_to_categorical
from deep4deep.text_processing import text_preprocessing, remove_special_chars, remove_punctuation
from deep4deep.w2v_embedding_and_rnn_model import Preprocessor, Embedder, LstmModel
from deep4deep import utils_w2v_rnn


def merge_df (df_from_main_module, my_df_meta_categorical):
    '''
    returns a tuple (df_with, df_without) where:
    df_with has only df_from_main_module rows that were present in my_df_meta_categorical (ie where text was retrieved) and all columns from my_df_meta_categorical
    df_without has only df_from_main_module rows that were absent from my_df_meta_categorical, and only the id columns)
    '''
    # both have id as col (not index)
    df_from_main_module = df_from_main_module[['id']] # keep no other col
    #import ipdb; ipdb.set_trace()
    df_with = df_from_main_module.merge(my_df_meta_categorical, on='id', how='inner', )
    return df_with


class LSTM_Meta_Trainer():

    def __init__(self):
        print("initializing Trainer")
        self.df_meta_categorical = data_prep()
        self.model = LstmModel()
        self.preprocessor = Preprocessor()
        print("downloading class transfer learning embedder – this may take time, coffee break maybe?")
        self.embedder = Embedder()

    @simple_time_tracker
    def lstm_training(self, train_set, val_set):
        '''
        train_set, val_set are received from the main module bpifrance_deeptech_analysis
        and are used to make:
        - our X through the lists of dealroom ids,
        - the corresponding y
        lstm_training:
        Trains the instance's model
        Returns the model
        '''
        train_with = merge_df(train_set, self.df_meta_categorical)
        val_with = merge_df(val_set, self.df_meta_categorical)

        #define my X
        X_train_ok = train_with['full_text']
        X_val_ok = val_with['full_text']
        # keeping a copy with all fields for classification errors examination
        X_val_check = val_with.copy().drop(columns=['target'])

        # define my y
        y_train_ok = train_with['target']
        y_val_ok = val_with['target']

        #prepare my X
        X_train_ok = self.preprocessor.fit_transform(X_train_ok)
        X_val_ok = self.preprocessor.transform(X_val_ok)

        X_train_ok = self.embedder.fit_transform(X_train_ok)
        X_val_ok = self.embedder.transform(X_val_ok)

        # fit the model
        print("fitting the model. this may take a few minutes.")
        self.model.fit(X_train_ok, y_train_ok, X_val_ok, y_val_ok)

        #provide feedback and metrics
        X_val_check = utils_w2v_rnn.make_X_check(X_val_ok, y_val_ok, X_val_check, self.model)
        utils_w2v_rnn.my_metrics(X_val_check)
        utils_w2v_rnn.rmse(X_val_ok, y_val_ok, self.model)
        utils_w2v_rnn.plot_loss_accuracy(self.model.history)

        return self.model

    @simple_time_tracker
    def lstm_predict(self, X_test_from_main_module):
        '''
        returns a dataframe of Dealroom ids and predictions of being deeptech, based on description text,
        in field 'y_pred',
        with NaN when there was no text to predict from
        '''

        # keep only those lines where I have text data
        X_test_with = merge_df(X_test_from_main_module, self.df_meta_categorical)
        # in real life, it won't be a merge like this, but a scraping and querying of data,
        # then dropping the lines where there is no data

        #define my X
        X_test_ok = X_test_with['full_text']

        #prepare my X
        X_test_ok = self.preprocessor.fit_transform(X_test_ok)
        X_test_ok = self.embedder.fit_transform(X_test_ok)

        # perform prediction on X_test
        X_test_with['y_pred'] = self.model.predict(X_test_ok)
        X_test_return = X_test_from_main_module.merge(X_test_with[['id','full_text', 'y_pred']], on='id', how='left')
        return X_test_return

    def save_as_joblib(target_file="deeptech_NLP_model.joblib"):
        '''
        save preferentially after a final training on all data available
        '''
        print(f"model will be saved as {target_file}")
        return joblib.dump(self.model, target_file)


def demo():
    # demo routine
    my_path = path.join(path.dirname(path.dirname(__file__)), "raw_data", "data2020-12-03.csv")

    df = pd.read_csv(my_path)

    recu_train, recu_test = train_test_split(df, test_size = .25)

    train_set, val_set = train_test_split(recu_train, test_size = .25)

    lstm_trainer = LSTM_Meta_Trainer()
    lstm_trainer.lstm_training(train_set, val_set)

    lstm_trainer.lstm_predict(recu_test.drop(columns=['target']))

    lstm_trainer.save_as_joblib()

    return lstm_trainer


if __name__ == '__main__':

    demo()

    # usage pour Cath:

    # me passer son X_train, y_train, X_val, y_val pour entraînement
    # me passer son X_test pour prédiction à "moyenner" avec la sienne
    # refaire son calcul de perf à partir de ça
