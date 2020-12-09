EMBEDDING = "glove-wiki-gigaword-300"

import gensim.downloader as api

embedding = api.load(EMBEDDING)
