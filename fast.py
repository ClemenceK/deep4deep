from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gensim.downloader as api
from tensorflow import keras
import pandas as pd

from deep4deep.w2v_embedding_and_rnn_model import Preprocessor, Embedder
from deep4deep.predictor import Predictor

from typing import Optional
from fastapi import Depends

from memoized_property import memoized_property

class AAA():
    @memoized_property
    def load_stuff(self):
        EMBEDDING = "glove-wiki-gigaword-300"

        import gensim.downloader as api
        import nltk

        embedding = api.load(EMBEDDING)
        nltk.download('stopwords')
        nltk.download('punkt')

        # créer une instance de Preprocessor()
        preprocessor = Preprocessor()
        # créer une instance de Embedder()
        # ceci prend plusieurs minutes
        embedder = Embedder()
        # créer une instance de prédicteur, qui charge le modèle
        predictor = Predictor(preprocessor, embedder, "deeptech_NLP_model")

        loaded = True

        commons = {}
        commons["loaded"] = loaded
        commons["preprocessor"] = preprocessor
        commons["embedder"] = embedder
        commons["predictor"] = predictor

        return commons

aaa = AAA()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


print("🍾 🍾 🍾 >>> APP created")

@app.get("/")
def index():
    print("🥰 🥰 🥰 >>> index called")

    return {"content": ["You've reached the homepage of NLP deeptech prediction API.",
                        "Prepare a dictionary of type: {'id_1':snippet_1, 'id_2':snippet_2}",
                        "where id is an identifier for the companies you want a prediction about, and description is a text describing the company",
                        "Then use the /pred endpoint"]}

@app.get("/load")
async def load(commons: dict = Depends(common_parameters)):
    print("🥰 🥰 🥰 >>> LOAD called")

    loaded = commons["loaded"]
    preprocessor = commons["preprocessor"]
    embedder = commons["embedder"]
    predictor = commons["predictor"]

    if loaded is False:
        print("😊 😊 😊 >>> LOADING WORD2VEC")
        # créer une instance de Preprocessor()
        preprocessor = Preprocessor()
        # créer une instance de Embedder()
        # ceci prend plusieurs minutes
        embedder = Embedder()
        # créer une instance de prédicteur, qui charge le modèle
        predictor = Predictor(preprocessor, embedder, "deeptech_NLP_model")

        loaded = True
    else:
        print("🤔 🤔 🤔 >>> WORD2VEC already loaded")

    commons["loaded"] = loaded
    commons["preprocessor"] = preprocessor
    commons["embedder"] = embedder
    commons["predictor"] = predictor

    return { "loaded" : loaded}

@app.post("/pred")
async def make_pred(content: dict):  # , commons: dict = Depends(common_parameters)):
    print("🥰 🥰 🥰 >>> MAKE PRED called")

    commons = aaa.load_stuff

    loaded = commons["loaded"]
    preprocessor = commons["preprocessor"]
    embedder = commons["embedder"]
    predictor = commons["predictor"]

    company_descriptions = content['company_descriptions']
    X_to_predict = pd.Series(company_descriptions.values())
    result = predictor.predict(X_to_predict)
    dict_to_return = dict(zip(company_descriptions.keys(),[float(i[0]) for i in result]))
    return dict_to_return
