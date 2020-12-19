import numpy
import string
import regex
import re
import unidecode
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

#from deep4deep.utils import simple_time_tracker



def remove_numbers(text):
    """
    removes numbers from text
    text: string
    returns text without numbers as a string
    """
    return ''.join(char for char in text if not char.isdigit())

def remove_punctuation(text):
    """
    removes punctuation from text
    text: string
    returns text without punctuation as a string
    """
    for punctuation in string.punctuation+"’":
        text = text.replace(punctuation, ' ')
        # adding ` as French apostrophe wasn't treated
        # and replacing by space to avoid "lapparition"
    return text

def remove_special_chars(text):
    return regex.sub(r'\p{So}+', ' ', text)

def remove_accents(text):
    return unidecode.unidecode(text)


def remove_stopwords(text):
    """
    removes stopwords from text
    text: string
    returns text without stopwords as a list of words
    """
    my_stopwords = set(stopwords.words('english'))
    my_stopwords.add('•')
    my_stopwords.add('’')
    #adding special characters found in hello tomorrow reports

    tokens = word_tokenize(text) # correspond à un split
    # also removing single characters
    tokens = [word for word in tokens if (len(word)>2 or word == "ai" or word == "ia")]
    # also removing 2 letter words except AI and IA (as there are French snippets so at least not le, la…)

    return [word for word in tokens if word not in my_stopwords]

def lemmatize(tokenized_text):
    """
    tokenized_text: list of words
    returns lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokenized_text]

def stem(tokenized_text):
    """
    tokenized_text: list of words
    returns stemmed text
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokenized_text]


def text_preprocessing(text):
    """
    applies preprocessing steps
    text: string
    returns preprocessed, tokenized text
    """
    try:
        text = text.lower()
    except: # exception thrown if NaN, None…
        print(f"text was {text}, replacing by empty string")
        return ""
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_special_chars(text)
    text = remove_accents(text)

    tokenized_text = remove_stopwords(text)

    #can add either stem or lemmatize

    return tokenized_text


# used in data preparation (as it needs the name from the Dealroom data)
def remove_own_name(text, name):
    return text.replace(name, "") #regex.sub(name, "", text)

#########################################################################

# unused
def dealroom_phrase_removal(text):
    dealroom_phrase = r"Here you'll find information about their funding, investors and team."
    if dealroom_phrase in text:
        text = re.sub(dealroom_phrase, "", text)
    return text
