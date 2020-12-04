
import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd

from deep4deep.utils import simple_time_tracker
from deep4deep.text_processing import text_preprocessing

import ast

@simple_time_tracker
def get_dealroom_meta_description(dealroom_url):
    '''
    given a website url (complete with http…),
    returns the description from the meta tags
    returns: string
    '''

    # no exception planned as all dealroom company pages have metadata
    response = requests.get(dealroom_url)
    soup = BeautifulSoup(response.content, "html.parser")
    description = soup.find("meta", property="og:description")["content"]

    return description

### TODO: tbd if always in english

@simple_time_tracker
def get_meta_description(row):
    '''
    given a website id,
    returns the description from the meta tags
    returns: string
    '''

    #essayer le site de l'entreprise, si ça ne marche pas "se rabattre" sur la page de dealroom
    website = row['website_url']
    try:
        response = requests.get(website)
        soup = BeautifulSoup(response.content, "html.parser")
        description = soup.find("meta", property="og:description")["content"]
    except:
        print(f"website {website} request threw an error, using dealroom page instead")
        description = get_dealroom_meta_description(row['url'])

    return description

def prepare_my_df(df):
    '''
    from raw data as provided by the extraction, returns a simpler dataframe
    with only useful columns
    '''
    NLP_useful_columns = ['id', 'name', 'industries','url', 'website_url', 'target']
    my_df = df[NLP_useful_columns].copy()
    my_df['industries'] = my_df['industries'].map(lambda x: re.findall(r".+?'name': '([^']+)", x))
    return my_df

def get_meta_description_columns(my_df, save_file_name="my_df_with_metatags.csv"):
    '''
    from a dataframe with ['id', 'name', 'industries','url', 'website_url', 'target'] columns, scrape
    companies' sites (or Dealroom page if the site sends back an error)
    returns a dataframe with new columns:
    ['meta_description']: text retrieved
    ['meta_description_preprocessed']: text preprocessed using text_processing.
    '''
    #essayer le site de l'entreprise, si ça ne marche pas "se rabattre" sur la page de dealroom
    my_df['meta_description'] = my_df.apply(get_meta_description, axis = 1) #apply row by row
    my_df['meta_description_dealroom'] = my_df['url'].map(get_dealroom_meta_description)

    my_df['meta_description_preprocessed'] = my_df['meta_description'].map(text_preprocessing)
    my_df['meta_description_dealroom_preprocessed'] = my_df['url'].map(text_preprocessing)

    my_df.to_csv(save_file_name)
    return my_df

def read_my_df_with_metatags_csv(file_name):

    my_df = pd.read_csv(file_name)
    my_df['meta_description_preprocessed'] = my_df['meta_description_preprocessed'].apply\
                            (lambda string: list(ast.literal_eval(string)))
    my_df['meta_description_dealroom_preprocessed'] = \
                            my_df['meta_description_dealroom_preprocessed'].apply\
                            (lambda string: list(ast.literal_eval(string)))
    return my_df


@simple_time_tracker
def import_dealroom_news(company_id=214127, n_news=1):
    '''
    queries the Dealroom API for news and returns a clean string, in English,
    comprising up to n_news most recent news.
    For each, are returned:
    title, summary, name of feed, content with all html tags removed, names of categories
    arguments:
    company_id: integer, Dealroom company id
    n_news: max quantity of news to retrieve
    returns: one single concatenated string, empty if the API returned an error
    '''
    URL = 'https://api.dealroom.co/api/v1'

    APIKEY = os.getenv('DEALROOMAPIKEY')
    if not APIKEY:
        print(f"For {company_id}, the API key could not be retrieved correctly")

    response = requests.get(
                    url = f'{URL}/companies/{company_id}/news',\
                    auth = (APIKEY, '')
                    )
    #import ipdb; ipdb.set_trace()
    return_string = [] #temporary a list

    try:
        print("im in try")
        json = response.json()
        for item in json['items']:
            return_string.append(item.get('title',""))
            return_string.append(item.get('summary',""))
            feed = item.get('feed',"")
            if type(feed) == dict:
                return_string.append(item.get('name',""))
            try:
                content = re.sub(r"<.+?>", '', item['content']) # remove all HTML tags from content
                return_string.append(content)
            except:
                print(f"No content to retrieve")
            try:
                categories_names = ' '.join([i['name'] for i in item['categories'] if i['name'] != None])
                return_string.append(categories_names)
            except:
                print(f"No categories.")
            while None in return_string: return_string.remove(None) # some categories are set to None, causing issues
        return " ".join(return_string)

    except:
        print("im in except")
        print(f"For {company_id}, the Dealroom news API returned no results")
        return ""
#TODO global news function

if __name__ == "__main__":
    print(import_dealroom_news(1598607))
