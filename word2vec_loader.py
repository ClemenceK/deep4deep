EMBEDDING = "glove-wiki-gigaword-300"

import gensim.downloader as api
import nltk

embedding = api.load(EMBEDDING)
nltk.download('stopwords')
nltk.download('punkt')
