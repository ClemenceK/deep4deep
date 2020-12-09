FROM python:3.8.6-buster

COPY req.txt /req.txt
RUN pip install -r req.txt

COPY fast.py /fast.py
COPY word2vec_loader.py /word2vec_loader.py

RUN python /word2vec_loader.py

CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
