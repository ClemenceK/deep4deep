from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gensim.downloader as api


EMBEDDING = "glove-wiki-gigaword-300"


embedding = api.load(EMBEDDING)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app = FastAPI()
@app.post("/pred")
def make_pred():
    print(len(embedding.wv.vocab))
    return {"pred": len(embedding.wv.vocab)}


