import pandas as pd
from sklearn.model_selection import train_test_split
from os import path


from deep4deep.text_retrieval import prepare_my_df, get_meta_description_columns,get_meta_description,get_dealroom_meta_description
from deep4deep.text_processing import text_preprocessing, remove_special_chars, remove_punctuation
from deep4deep.w2v_embedding_and_rnn_model import change_to_categorical, Preprocessor, Embedder, LstmModel
from deep4deep import utils_w2v_rnn
from deep4deep.utils import simple_time_tracker



@simple_time_tracker
def data_prep(df=None):
   # data preparation routine
#===============================================================================
    # If no existing file

        # use a raw_data file
        # df = pd.read_csv('../raw_data/data2020-12-03.csv')
        # my_df = prepare_my_df(df)

        # this is to scrape the data:
        #my_df_meta = get_meta_description_columns(my_df)
        # avoid redoing it every time (takes 1 hour +)
        # instead let's take the existing one, already scraped:

#===============================================================================
    # If there is an existing file

        # you need a file with metadata (obtained by scraping using the text_retrieval functions)
        # in a 'raw_data' folder st the same level as the deep4deep package
        path_df_meta = path.join(path.dirname(path.dirname(__file__)),"raw_data", "my_df_with_metatags.csv")
        my_df_meta = pd.read_csv(path_df_meta)

        # concaténer dealroom_meta_description et meta_description
        my_df_meta['full_text'] = my_df_meta.dealroom_meta_description + " " + my_df_meta.meta_description

        # dropping "almost_deep_tech"
        my_df_meta_categorical = change_to_categorical(my_df_meta)
        return my_df_meta_categorical

@simple_time_tracker
def merge_df (df_from_main_module, my_df_meta_categorical):
    # both have id as col (not index)
    df_from_main_module = df_from_main_module[['id']] # keep no other col
    # inner -> keep only the rows that are in both (i.e. selected in df_from_main_module,
    # and with data in )
    #import ipdb; ipdb.set_trace()
    return df_from_main_module.merge(my_df_meta_categorical, on='id', how='inner', )


class LSTM_Meta_Trainer():

    def __init__(self):
        print("initializing Trainer")
        self.df_meta_categorical = data_prep()
        self.preprocessor = Preprocessor()
        print("downloading transfer learning embedder – this may take time, coffee break maybe?")
        self.embedder = Embedder()
        self.model = LstmModel()

    @simple_time_tracker
    def lstm_training(self, X_train_main_module, X_val_main_module, y_train, y_val):
        '''
        X_train and X_val are only used to get their lists of dealroom ids,
        that allow to reconstitute our X.
        Trains the instance's model
        Returns the model
        '''
        #define my X
        train = merge_df(X_train_main_module, self.df_meta_categorical)
        val = merge_df(X_val_main_module, self.df_meta_categorical)

        X_train = train['full_text']
        X_val = val['full_text']
        self.X_val_check = X_val.copy()


        #prepare my X
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)

        X_train = self.embedder.fit_transform(X_train)
        X_val = self.embedder.transform(X_val)

        # fit the model
        print("fitting the model. this may take a few minutes.")
        self.model.fit(X_train, y_train, X_val, y_val)

        #provide feedback and metrics
        self.X_val_check = utils_w2v_rnn.make_X_check(X_val, y_val, X_val_check, model)
        utils_w2v_rnn.my_metrics(self.X_val_check)
        utils_w2v_rnn.rmse(X_val, y_val, model)
        utils_w2v_rnn.plot_loss_accuracy(model.history)

        return model

    @simple_time_tracker
    def lstm_predict(X_test_main_module):
        #define my X
        test = merge_df(X_test_main_module, self.df_meta_categorical)

        X_test = test['full_text']
        self.model.predict(X_test)

    #TODO

def demo():
    # demo routine
    my_path = path.join(path.dirname(path.dirname(__file__)), "raw_data", "data2020-12-03.csv")

    recu_train, recu_test = train_test_split(read_csv(my_path), test_size = .25)

    train, test = train_test_split(recu_train, test_size = .25)
    X_train_main_module = train.drop(columns=['target'])
    y_train=train['target']
    X_val_main_module = train.drop(columns=['target'])
    y_val=train['target']

    lstm_trainer = LSTM_Meta_Trainer()
    lstm_trainer.lstm_training(X_train_main_module, X_val_main_module, y_train,y_val)

    return lstm_trainer

if __name__ == '__main__':

    demo()



# usage pour Cath:

# me passer son df_train (ou X_train, y_train) pour entraînement
# me passer son df_test (ou …) pour prédiction à "moyenner" avec la sienne
# refaire son calcul de perf à partir de ça
