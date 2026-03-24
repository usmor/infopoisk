import numpy as np
import pickle
import os
from typing import List, Dict
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText
import sys
sys.path.append(os.path.dirname(__file__))
from base_index import BaseIndex  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEXES_DIR = os.path.join(BASE_DIR, "../indexes")


class FastTextIndex(BaseIndex):
    """
    Индекс на основе FastText.

    Обучает модель Word2Vec на лемматизированном корпусе и создает
    векторные представления документов путем усреднения векторов слов.
    """

    def __init__(self,
                 vector_size: int = 300,
                 window: int = 5,
                 min_count: int = 2,
                 workers: int = 4,
                 epochs: int = 10,
                 min_n: int = 3,
                 max_n: int = 6,
                 **kwargs):
        """
        Args:
            vector_size: Размерность векторов
            window: Размер окна контекста
            min_count: Минимальная частота слова
            workers: Количество потоков
            epochs: Количество эпох
            min_n: Минимальная длина char n-gram
            max_n: Максимальная длина char n-gram
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.min_n = min_n
        self.max_n = max_n
        self.kwargs = kwargs

        self.model = None
        self.doc_vectors = None
        self.lemmatized_corpus = []
        self.mapping = []
        self.talk_languages = {}

        os.makedirs(INDEXES_DIR, exist_ok=True)

        self.model_path = os.path.join(INDEXES_DIR, 'fasttext_trained.model')
        self.vectors_path = os.path.join(
            INDEXES_DIR, 'fasttext_trained_vectors.npy')
        self.mapping_path = os.path.join(
            INDEXES_DIR, 'fasttext_trained_mapping.pkl')
        self.languages_path = os.path.join(
            INDEXES_DIR, 'fasttext_trained_languages.pkl')

    def build(self, data: List[Dict]):
        """
        Построение FastText индекса.
        """
        for item in tqdm(data, desc="Подготовка документов"):
            talk_id = item['talk_id']
            transcripts = item['lemmatized_transcripts']

            self.talk_languages[talk_id] = list(transcripts.keys())

            for lang, text in transcripts.items():
                lemmas = text.split()
                self.lemmatized_corpus.append(lemmas)
                self.mapping.append({
                    'talk_id': talk_id,
                    'language': lang
                })

        self.model = FastText(
            sentences=self.lemmatized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            min_n=self.min_n,
            max_n=self.max_n,
            **self.kwargs
        )

        doc_vectors = []
        for tokens in tqdm(self.lemmatized_corpus, desc="Усреднение векторов"):
            vectors = []
            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])

            if vectors:
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(self.vector_size))

        self.doc_vectors = np.array(doc_vectors)

    def save(self):
        """
        Сохраняет модель и метаданные.
        """
        self.model.save(self.model_path)
        np.save(self.vectors_path, self.doc_vectors)

        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.mapping, f)

        with open(self.languages_path, 'wb') as f:
            pickle.dump(self.talk_languages, f)

    def load(self):
        """
        Загружает модель и метаданные.
        """
        self.model = FastText.load(self.model_path)
        self.doc_vectors = np.load(self.vectors_path)

        with open(self.mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)

        with open(self.languages_path, 'rb') as f:
            self.talk_languages = pickle.load(f)

    def search(self,
               query_lemmas: List[str],
               top_k: int = 5,
               query_language: str = None):
        """
        Поиск по запросу с использованием косинусной близости.

        Args:
            query_lemmas: Токенизированный и лемматизированный запрос
            top_k: Количество результатов
            query_language: Язык запроса для усиления
        """

        if self.model is None or self.doc_vectors is None:
            self.load()

        query_vector = self.get_query_vector(query_lemmas)

        if query_vector is None:
            return []

        similarities = cosine_similarity(
            query_vector.reshape(1, -1), self.doc_vectors)[0]

        results = []
        talk_ids_seen = set()
        for idx, score in enumerate(similarities):
            if score > 0:
                talk_id = self.mapping[idx]['talk_id']
                language = self.mapping[idx]['language']

                if language != query_language:
                    continue
                if talk_id not in talk_ids_seen:
                    talk_ids_seen.add(talk_id)
                    results.append((talk_id, float(score), language))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_query_vector(self, query_lemmas: List[str]):
        """
        Создает вектор запроса путем усреднения векторов слов.
        """
        vectors = []

        for lemma in query_lemmas:
            if lemma in self.model.wv:
                vectors.append(self.model.wv[lemma])

        if not vectors:
            return None

        return np.mean(vectors, axis=0)
