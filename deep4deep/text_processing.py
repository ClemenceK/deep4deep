import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer



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
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def remove_stopwords(text):
    """
    removes stopwords from text
    text: string
    returns text without stopwords as a string
    """
    my_stopwords = set(stopwords.words('english'))
    my_stopwords.add('•')
    my_stopwords.add('’')
    #adding special characters found in hello tomorrow reports

    tokens = word_tokenize(text) # correspond à un split
    return [word for word in tokens if word not in my_stopwords]

def lemmatize(text):
    """
    text: string
    returns lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

def text_preprocessing(text):
    """
    applies preprocessing steps
    text: string
    returns preprocessed, tokenized text
    """
    text = text.lower()
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)

    return lemmatize(text)
