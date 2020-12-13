FROM python:3.8.6-buster

RUN pip install --upgrade pip
COPY requirements_docker.txt /requirements_docker.txt
RUN pip install -r requirements_docker.txt

# COPY word2vec_loader.py /word2vec_loader.py
# RUN python /word2vec_loader.py

COPY fast.py /fast.py
COPY deep4deep/predictor.py /deep4deep/predictor.py
COPY deep4deep/w2v_embedding_and_rnn_model.py /deep4deep/w2v_embedding_and_rnn_model.py
COPY deep4deep/text_processing.py /deep4deep/text_processing.py
COPY deep4deep/__init__.py /deep4deep/__init__.py
COPY deeptech_NLP_model /deeptech_NLP_model

CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
