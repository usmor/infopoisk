from rank_bm25 import BM25Okapi
from typing import List, Dict
from tqdm.auto import tqdm
import pickle
import os
import sys
sys.path.append(os.path.dirname(__file__))
from base_index import BaseIndex  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEXES_DIR = os.path.join(BASE_DIR, "../indexes")


class BM25Index(BaseIndex):
    """
    Обратный индекс BM25 строится через библиотеку rank-bm25.
    """

    def __init__(self, k1=1.5, b=0.75, **kwargs):
        self.bm25_type = "okapi"
        self.k1 = k1
        self.b = b
        self.kwargs = kwargs

        self.retriever = None

        self.lemmatized_corpus = []
        self.mapping = []       # позиция -> (talk_id, язык)
        self.talk_languages = {}

        os.makedirs(INDEXES_DIR, exist_ok=True)

        self.index_path = os.path.join(INDEXES_DIR, 'bm25_index.pkl')
        self.mapping_path = os.path.join(INDEXES_DIR, 'bm25_mapping.pkl')
        self.languages_path = os.path.join(INDEXES_DIR, 'bm25_languages.pkl')

    def build(self, data: List[Dict]):
        """
        Построение BM25 индекса.
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

        self.retriever = BM25Okapi(
            self.lemmatized_corpus, k1=self.k1, b=self.b, **self.kwargs)

    def save(self):
        """
        Сохраняем индекс и параметры.
        """
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.retriever, f)
        print("bm25_index.pkl сохранён")

        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.mapping, f)
        print("bm25_mapping.pkl сохранён")

        with open(self.languages_path, 'wb') as f:
            pickle.dump(self.talk_languages, f)
        print("bm25_languages.pkl сохранён")

    def load(self):
        """
        Загружаем индекс и параметры.
        """

        with open(self.index_path, 'rb') as f:
            self.retriever = pickle.load(f)

        with open(self.mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)

        with open(self.languages_path, 'rb') as f:
            self.talk_languages = pickle.load(f)

    def search(self,
               query_lemmas: List[str],
               top_k: int = 5,
               query_language: str = None):
        """
        Поиск по запросу.
        Args:
            query: Текст запроса (или уже токены)
            top_k: Количество результатов
            query_language: Язык запроса (ru/en/fr)
        """

        if self.retriever is None:
            self.load()

        if isinstance(query_lemmas, str):
            query_lemmas = query_lemmas.split()
        else:
            query_lemmas = query_lemmas

        scores = self.retriever.get_scores(query_lemmas)

        results = []
        talk_ids_seen = set()
        for idx, score in enumerate(scores):
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
