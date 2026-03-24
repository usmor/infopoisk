import json
import os
from fasttext_index import FastTextIndex
from bm25_index import BM25Index
from word2vec_index import Word2VecIndex
import time

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PREPROCCESSED_DATA_PATH = os.path.join(
    BASE_DIR, "../data", "ted_preprocessed.json")

with open(PREPROCCESSED_DATA_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

print("Построение индекса BM25...")  # ~ 8-10 секунд
start_time = time.time()
bm25 = BM25Index(k1=1.5, b=0.75)
bm25.build(documents)
bm25.save()
print("Индекс BM25 построен и сохранен")
print(
    f"Время построения и сохранения индекса BM25: {round(time.time() - start_time, 4)} сек")

print("Построение индекса Word2Vec...")  # ~ 2-2.5 минуты
start_time = time.time()
word2vec = Word2VecIndex(vector_size=200, window=5,
                         min_count=2, workers=4, epochs=10)
word2vec.build(documents)
word2vec.save()
print("Индекс Word2Vec построен и сохранен")
print(
    f"Время построения и сохранения индекса Word2Vec: {round(time.time() - start_time, 4)} сек")

print("Построение индекса Fasttext...")  # ~ 6-7 минут
start_time = time.time()
fasttext = FastTextIndex(vector_size=100, window=5, min_count=2,
                         workers=4, epochs=10)
fasttext.build(documents)
fasttext.save()
print("Индекс Fasttext построен и сохранен")
print(
    f"Время построения и сохранения индекса Fasttext: {round(time.time() - start_time, 4)} сек")
