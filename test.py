
EMBEDDING = "glove-wiki-gigaword-300"

import gensim.downloader as api

embedding = api.load(EMBEDDING)

print(len(embedding.wv.vocab))
