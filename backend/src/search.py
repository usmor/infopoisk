import time
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from backend.src.data_storage import DataStorage  # noqa: E402
from backend.src.preprocessing import lemmatize_for_query  # noqa: E402
from backend.src.language_detector import language_detector  # noqa: E402


storage = DataStorage()


class SearchEngine:
    def __init__(self):
        self.storage = DataStorage()

        self.indexes = {}
        self._loaded_indexes = set()

    def _load_index(self, index_type):
        """
        Загружает конкретный индекс по требованию.
        """

        if index_type in self._loaded_indexes:
            return

        if index_type == "bm25":
            from backend.src.build_indexes.bm25_index import BM25Index
            idx = BM25Index(k1=1.5, b=0.75)
            idx.load()
        elif index_type == "word2vec":
            from backend.src.build_indexes.word2vec_index import Word2VecIndex
            idx = Word2VecIndex(vector_size=300, window=5, min_count=2)
            idx.load()
        elif index_type == "fasttext":
            from backend.src.build_indexes.fasttext_index import FastTextIndex
            idx = FastTextIndex(vector_size=300, window=5, min_count=2)
            idx.load()

        self.indexes[index_type] = idx
        self._loaded_indexes.add(index_type)

    def search(self, query: str, index_type: str = "bm25", top_k: int = 10):
        start_time = time.time()
        if not query.strip():
            raise ValueError("Запрос не может быть пустым.")

        if not index_type:
            raise ValueError("Тип индекса не может быть пустым.")
        if index_type not in ["bm25", "word2vec", "fasttext"]:
            raise ValueError("Недопустимый тип индекса.")

        self._load_index(index_type)
        index = self.indexes[index_type]

        query_lang = language_detector(query)
        query_lemmas = lemmatize_for_query(query, query_lang)

        raw_results = index.search(
            query_lemmas, top_k=top_k, query_language=query_lang)

        enriched_results = []
        for talk_id, score, lang in raw_results:
            available_langs = self.storage.get_available_languages(talk_id)
            description = self.storage.get_description(talk_id, query_lang)
            enriched_results.append({
                'talk_id': talk_id,
                'score': score,
                'language': lang,
                'url': self.storage.get_url(talk_id),
                'title': self.storage.get_title(talk_id, query_lang),
                'speakers': self.storage.get_speakers(talk_id),
                'description': description,
                'transcript': self.storage.get_transcript(talk_id, query_lang),
                'available_languages': available_langs
            })

        search_time = time.time() - start_time

        return {
            'query': query,
            'query_lang': query_lang,
            'index_type': index_type,
            'results': enriched_results,
            'total_found': len(enriched_results),
            'search_time': round(search_time, 4)
        }
