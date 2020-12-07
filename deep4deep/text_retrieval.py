
import requests
from bs4 import BeautifulSoup
import re
import os
import os.path
import pandas as pd

from deep4deep.utils import simple_time_tracker
from deep4deep.text_processing import text_preprocessing, remove_own_name
from deep4deep.utils import simple_time_tracker


import ast

@simple_time_tracker
def get_dealroom_meta_description(row):
    '''
    given the dealroom page url (complete with http…),
    returns the description from the meta tags
    returns: the replacement row
    '''

    # take tagline and remove the company name
    description = remove_own_name(row['tagline'], row['name'])
    return description # to be added in 'dealroom_meta_description' column

@simple_time_tracker
def get_meta_description(row):
    '''
    given a my_df row,
    scraps the description from the meta tags in website_url
    returns: the replacement row
    warning: use after creating column 'dealroom_meta_description'
    '''

    website = row['website_url']
    try:
        response = requests.get(website)
        soup = BeautifulSoup(response.content, "html.parser")
        description = soup.find("meta", property="og:description")["content"]
        description = remove_own_name(description, row['name'])
    except:
        print(f"website {website} request threw an error, imputing dealroom_meta_description instead")
        description = row['dealroom_meta_description']
    return description # to be added in 'meta_description' column

@simple_time_tracker
def prepare_my_df(df):
    '''
    from raw data as provided by the extraction, returns a simpler dataframe
    with only useful columns
    '''

    NLP_useful_columns = ['id', 'name', 'tagline', 'url', 'website_url', 'target']
    my_df = df[NLP_useful_columns].copy()

    my_df.set_index('id', inplace=True)

    # drop a problematic line (the first one: scrapings returns junk text instead of error;
    # for all others, just no or French data but no big deal if they stay)
    problematic_ids_to_drop = [969633, 971808, 31373, 1742840, 1660559, 1836530,
    227608, 933434, 1831991,1834603, 217428,1836415, 1742840, 1834791, 1466670,
    1817120, 1836255, 1836503,1921970,1891276, 906637, 198955, 1738965, 1855449,
    1787891, 1800559, 1836943, 1834666, 1835159, 1835167, 1835172, 1801785, 1836114,
    1836371, 1836433, 1836530, 1832864, 968692, 1836470, 1836474, 894898, 908848,
    1836732, 1463619, 144370, 1836822, 1800595, 1837085, 1837166, 1987283]

    for i in problematic_ids_to_drop:
        try:
            my_df.drop(index=[], inplace=True)
        except:
            print(f"index {i}, which we usually drop for quality issues, is not present in dataset (which is ok)")

    #my_df['industries'] = my_df['industries'].map(lambda x: re.findall(r".+?'name': '([^']+)", x))
    return my_df

@simple_time_tracker
def get_meta_description_columns(my_df, save_file_name="my_df_with_metatags.csv"):
    '''
    from a dataframe with ['name', 'tagline', website_url'] columns at least,
    process dealroom tagline THEN scrape companies pages for meta description
    returns a dataframe with new str columns:
    ['dealroom_meta_description']
    ['meta_description']
    '''
    my_df.loc[:,'dealroom_meta_description'] = my_df.tag.apply(get_dealroom_meta_description, axis = 1) #apply row by row
    my_df.loc[:,'meta_description'] = my_df.apply(get_meta_description, axis = 1)
    my_df.to_csv(save_file_name)
    return my_df

# launch as a module to make the dataframe
if __name__ == '__main__':
    path = path.join(path.dirname(path.dirname(__file__)), "raw_data", "data2020-12-03.csv")
    df = pd.read_csv(path)
    my_df = prepare_my_df(df)
    my_df_with_metatags = get_meta_description_columns(my_df)






#####################################################################################
#ununsed

def read_my_df_with_metatags_csv(file_name):
    # use only if you saved a df with prepreprocessed columns – change col names as needed
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
