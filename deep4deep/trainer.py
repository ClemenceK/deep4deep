import pandas as pd
from sklearn.model_selection import train_test_split
from os import path
from tensorflow import keras



from deep4deep.utils import simple_time_tracker
from deep4deep.text_retrieval import prepare_my_df, get_meta_description_columns,get_meta_description,get_dealroom_meta_description
from deep4deep.description_dataframe_preparation import data_prep, change_to_categorical
from deep4deep.text_processing import text_preprocessing
from deep4deep.w2v_embedding_and_rnn_model import Preprocessor, Embedder, LstmModel
from deep4deep import utils_w2v_rnn


def merge_df (df_from_main_module, my_df_meta_categorical):
    '''
    returns df_with where:
    df_with has only df_from_main_module rows that were present in my_df_meta_categorical (ie where text was retrieved) and all columns from my_df_meta_categorical
    '''
    # both have id as col (not index)
    df_from_main_module = df_from_main_module[['id']] # keep no other col
    #import ipdb; ipdb.set_trace()
    df_with = df_from_main_module.merge(my_df_meta_categorical, on='id', how='inner', )
    return df_with


class LSTM_Meta_Trainer():

    def __init__(self, preprocessor, embedder):
        print("initializing Trainer")
        self.df_meta_categorical = data_prep()
        self.model = LstmModel()
        self.preprocessor = preprocessor
        self.embedder = embedder

    @simple_time_tracker
    def lstm_training(self, train_set, val_set):
        '''
        train_set, val_set are typically received from main module bpifrance_deeptech_analysis
        and are used to build:
        - our X through the lists of dealroom ids, using our text data,
        - the corresponding y
        val_set is used it for early stopping and to print scoring evaluations
        The function trains the instance's model
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
        #utils_w2v_rnn.rmse(X_val_ok, y_val_ok, self.model)
        utils_w2v_rnn.plot_loss_accuracy(self.model.history)

        return self.model

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

    def save_model(self, target_file="deeptech_NLP_model"):
        '''
        save preferentially after a final training on all data available
        '''
        return self.model.model.save(target_file)
        #to later load: model = keras.models.load_model('path/to/model')

def main_trainer():
    '''
    Trains a model by reading the ./raw_data/data2020-12-03.csv dataset
    (a file from the main model, with Dealroom data extracted through the API)
    using its own set of text data (./raw_data/)
    based on the my_df_with_metatags.csv file.
    Can be used to re-train the model on an updated set of data.
    Alternatively, use the "training_the_model" notebook
    '''

    # data preparation
    my_path = path.join(path.dirname(path.dirname(__file__)), "raw_data", "data2020-12-03.csv")
    df = pd.read_csv(my_path)
    # using minimum val size here, just to enable Early_stopping,
    # as hyperparameters tuning has already been done.
    train, val = train_test_split(df, test_size = .10)


    # instanciations
    preprocessor = Preprocessor()
    embedder = Embedder()
    lstm_trainer = LSTM_Meta_Trainer(preprocessor, embedder)


    lstm_trainer.lstm_training(train, val)

    # saving the trained model
    lstm_trainer.save_model()

    return lstm_trainer


def from_Cathsfile_to_mine(my_path, i=""):
    # data preparation
    X_train_set = pd.read_csv(my_path+"X_test"+i+".csv", index_col=0)
    y_train_set = pd.read_csv(my_path+"y_test"+i+".csv", index_col=0)

    train_set = X_train_set.copy()
    train_set['target'] = y_train_set

    X_val_set = pd.read_csv(my_path+"X_test"+i+".csv", index_col=0)
    y_val_set = pd.read_csv(my_path+"y_test"+i+".csv", index_col=0)

    val_set = X_val_set.copy()
    val_set['target'] = y_val_set
    return train_set, val_set

def preparing_results():
    '''
    Prepares the files results_1, results_2, results_3 for scoring
    thrice the combination of the two models (main model + NLP model)
    as done in the notebook Scoring_the_models.ipynb
    Used just once to prepare those files.
    '''
    my_path = path.join(path.dirname(path.dirname(__file__)), "raw_data", "data_cross_val", "")

    train_set_1, val_set_1 = from_Cathsfile_to_mine(my_path, "")
    train_set_2, val_set_2 = from_Cathsfile_to_mine(my_path, "2")
    train_set_3, val_set_3 = from_Cathsfile_to_mine(my_path, "3")

    # instanciations
    preprocessor = Preprocessor()
    embedder = Embedder()
    lstm_trainer = LSTM_Meta_Trainer(preprocessor, embedder)

    #1
    train_train_set_1, early_stop_val_set_1 = train_test_split(train_set_1, test_size = .2)
    lstm_trainer.lstm_training(train_train_set_1, early_stop_val_set_1)

    results_1 = lstm_trainer.lstm_predict(val_set_1)[['id', 'name', 'full_text', 'target','y_pred']]
    results_1.rename(columns={'y_pred':'y_pred_NLP'}, inplace=True)

    results_1 = val_set_1[['id', 'name', 'target']].merge(results_1[['id', 'y_pred_NLP']], on='id', how='left')
    # saving for score evaluation
    results_1.to_csv(my_path+"results_1.csv", index=False)

    #2
    train_train_set_2, early_stop_val_set_2 = train_test_split(train_set_2, test_size = .2)
    lstm_trainer.lstm_training(train_train_set_2, early_stop_val_set_2)

    results_2 = lstm_trainer.lstm_predict(val_set_2)[['id', 'name', 'full_text', 'target','y_pred']]
    results_2.rename(columns={'y_pred':'y_pred_NLP'}, inplace=True)
    results_2 = val_set_2[['id', 'name', 'target']].merge(results_2[['id', 'y_pred_NLP']], on='id', how='left')
    # saving for score evaluation
    results_2.to_csv(my_path+"results_2.csv",index=False)

    #3
    train_train_set_3, early_stop_val_set_3 = train_test_split(train_set_3, test_size = .3)
    lstm_trainer.lstm_training(train_train_set_3, early_stop_val_set_3)

    results_3 = lstm_trainer.lstm_predict(val_set_3)[['id', 'name', 'full_text', 'target','y_pred']]
    results_3.rename(columns={'y_pred':'y_pred_NLP'}, inplace=True)
    results_3 = val_set_3[['id', 'name', 'target']].merge(results_3[['id', 'y_pred_NLP']], on='id', how='left')
    # saving for score evaluation
    results_3.to_csv(my_path+"results_3.csv",index=False)

    return results_1, results_2, results_3


if __name__ == '__main__':

    main_trainer()
