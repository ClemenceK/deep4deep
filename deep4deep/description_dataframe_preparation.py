
import pandas as pd
from os import path

from deep4deep.utils import simple_time_tracker


#===============================================================================
    # If no existing file:

        # first use a raw_data file from dealroom scraping in bpifrance_deeptech_analysis module
        # df = pd.read_csv('../raw_data/data2020-12-03.csv')
        # my_df = prepare_my_df(df)

        # then use the "text_retrieval" routine from __main__ to scrape data
        # avoid redoing it every time (it takes 1 hour +)
        # instead let's take the existing one, already scraped
        # and saved by the get_meta_description_columns inside the text_retrieval __main__
        # under the name "my_df_with_metatags.csv"

        # then use data_prep on that file



#===============================================================================
    # If there is an existing file, use data_prep on that file



def change_to_categorical(my_df):
    my_df_categorical = my_df[my_df['target']!=0.5].copy()
    return my_df_categorical

def data_prep(file_path = path.join(path.dirname(path.dirname(__file__)),"raw_data", "my_df_with_metatags.csv")):
    '''
    file_path: path to a file with descriptions/metadata (obtained by the above described process)
    usually stored in a 'raw_data' folder at the same level as the deep4deep package
    '''

    my_df_meta = pd.read_csv(file_path)

    # NaN + " informations" donne Nan…
    # donc il faut gérer les NaN avant de faire le full text
    my_df_meta = my_df_meta.fillna({"dealroom_meta_description":"","meta_description":"" })
    # concaténer dealroom_meta_description et meta_description en full_text
    my_df_meta['full_text'] = my_df_meta.dealroom_meta_description + " " + my_df_meta.meta_description
    # ramener toutes les strings avec seulement des espaces à la string vide
    my_df_meta['full_text'] = my_df_meta['full_text'].map(lambda x : x.strip())
    # enlever les rows qui n'ont qu'une string vide dans "full_text"
    my_df_meta = my_df_meta[my_df_meta.full_text!=""]
    # enlever les rows qui ont nan dans full_text
    my_df_meta = my_df_meta.dropna(axis=0, subset=['full_text'])

    # making sure 'id' is int, as it is in the dataframe passed from the main model
    my_df_meta['id'] = my_df_meta.id.map(int)

    # drop "almost_deep_tech", pour n'avoir que 0 ou 1 en target
    my_df_meta_categorical = change_to_categorical(my_df_meta)

    return my_df_meta_categorical


    # changer si besoin les id en int (not object)
    #my_df_meta['id']=my_df_meta.id.map(int)


