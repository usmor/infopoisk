import numpy as np
import pickle
import os
from typing import List, Dict
from tqdm.auto import tqdm
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(os.path.dirname(__file__))
from base_index import BaseIndex  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEXES_DIR = os.path.join(BASE_DIR, "../indexes")
PREPROCCESSED_DATA_PATH = os.path.join(
    BASE_DIR, "../data", "ted_preprocessed.json")


class Word2VecIndex(BaseIndex):
    """
    Индекс на основе Word2Vec.

    Обучает модель Word2Vec на лемматизированном корпусе и создает
    векторные представления документов путем усреднения векторов слов.
    """

    def __init__(self,
                 vector_size: int = 300,
                 window: int = 5,
                 min_count: int = 2,
                 workers: int = 4,
                 epochs: int = 10,
                 **kwargs):
        """
        Args:
            vector_size: Размерность векторов
            window: Размер окна контекста
            min_count: Минимальная частота слова для включения в словарь
            workers: Количество потоков для обучения
            epochs: Количество эпох обучения
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.kwargs = kwargs

        self.model = None
        self.doc_vectors = None

        self.lemmatized_corpus = []
        self.mapping = []             # позиция -> (talk_id, язык)
        self.talk_languages = {}

        os.makedirs(INDEXES_DIR, exist_ok=True)

        self.model_path = os.path.join(INDEXES_DIR, 'word2vec_model.model')
        self.vectors_path = os.path.join(INDEXES_DIR, 'word2vec_vectors.npy')
        self.mapping_path = os.path.join(INDEXES_DIR, 'word2vec_mapping.pkl')
        self.languages_path = os.path.join(
            INDEXES_DIR, 'word2vec_languages.pkl')

    def build(self, data: List[Dict]):
        """
        Построение Word2Vec индекса.
        """
        for item in tqdm(data, desc="Обработка документов"):
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

        self.model = Word2Vec(
            sentences=self.lemmatized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            **self.kwargs
        )

        self.doc_vectors = self.create_doc_vectors()

    def create_doc_vectors(self):
        """
        Создает вектор для каждого документа путем усреднения векторов слов.
        """
        doc_vectors = []

        for tokens in tqdm(self.lemmatized_corpus, desc="Усреднение векторов"):
            vectors = []

            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])

            if vectors:
                doc_vector = np.mean(vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)

            doc_vectors.append(doc_vector)

        return np.array(doc_vectors)

    def save(self):
        """
        Сохраняет модель, векторы и метаданные.
        """
        self.model.save(self.model_path)

        np.save(self.vectors_path, self.doc_vectors)

        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.mapping, f)

        with open(self.languages_path, 'wb') as f:
            pickle.dump(self.talk_languages, f)

    def load(self):
        """
        Загружает модель, векторы и метаданные.
        """
        self.model = Word2Vec.load(self.model_path)

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
