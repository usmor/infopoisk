from whoosh.qparser import QueryParser
from whoosh import index
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import RegexTokenizer
from collections import defaultdict, Counter
from typing import Dict, List
import re
import pickle
import os
import shutil
import sys
from scipy.sparse import lil_matrix, save_npz, load_npz
sys.path.append(os.path.dirname(__file__))
from base_index import BaseIndex   # noqa: E402


class FrequencyIndexLib(BaseIndex):
    """
    Частотный обратный индекс через библиотеку Whoosh.
    Индекс сохраняется на диск.
    """

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.inverted_index: Dict[str, Dict[int, int]] = {}

    def build(self, texts: List[str]):
        """Строит индекс и сохраняет его на диск."""
        if os.path.exists(self.index_dir):
            shutil.rmtree(self.index_dir)
        os.mkdir(self.index_dir)

        schema = Schema(
            doc_id=ID(stored=True),
            content=TEXT(stored=True, analyzer=RegexTokenizer())
        )

        ix = create_in(self.index_dir, schema)
        writer = ix.writer()

        for doc_id, text in enumerate(texts):
            writer.add_document(
                doc_id=str(doc_id),
                content=text
            )

        writer.commit()

    def save(self, path: str):
        """Сохраняет словарь обратного индекса на диск."""
        with open(path, "wb") as f:
            pickle.dump(self.inverted_index, f)

    def load(self, path: str):
        """Загружает словарь обратного индекса с диска."""
        with open(path, "rb") as f:
            self.inverted_index = pickle.load(f)

    def search(self, query: str, top_k: int = 5):
        """Ищет документы, релевантные запросу, используя Whoosh."""
        if not os.path.exists(self.index_dir):
            raise ValueError(f"Индекс не найден в {self.index_dir}")

        ix = index.open_dir(self.index_dir)
        results_list = []

        with ix.searcher() as searcher:
            parser = QueryParser("content", ix.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=top_k)

            for hit in results:
                doc_id = int(hit['doc_id'])
                score = hit.score
                results_list.append((doc_id, score))

        return results_list


class FrequencyIndexDict(BaseIndex):
    """
    Частотный обратный индекс (словарь):
        слово -> {doc_id: частота}
    """

    def __init__(self):
        self.inverted_index: Dict[str, Dict[int, int]] = {}

    def build(self, texts: List[str]):
        """
        Строит обратный индекс из списка документов.
        """
        inverted_index = defaultdict(dict)

        for doc_id, text in enumerate(texts):
            tokens = re.findall(r"\w+|[^\w\s]", text)
            term_freq = Counter(tokens)

            for token, freq in term_freq.items():
                inverted_index[token][doc_id] = freq

        self.inverted_index = dict(inverted_index)

    def save(self, path: str):
        """
        Сохраняет объект индекса на диск.
        :param path: путь к файлу
        """
        with open(path, "wb") as f:
            pickle.dump(self.inverted_index, f)

    def load(self, path: str) -> "FrequencyIndexDict":
        """Загружает индекс с диска."""
        with open(path, "rb") as f:
            self.inverted_index = pickle.load(f)

    def search(self, query: str, top_k: int = 5):
        tokens = re.findall(r"\w+|[^\w\s]", query)
        scores = {}

        for token in tokens:
            if token in self.inverted_index:
                for doc_id, freq in self.inverted_index[token].items():
                    scores[doc_id] = scores.get(doc_id, 0) + freq

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class FrequencyIndexMatrix(BaseIndex):
    """
    Частотный обратный индекс в виде разреженной матрицы.
    """

    def __init__(self):
        self.term_to_idx: Dict[str, int] = {}
        self.matrix = None
        self.num_docs = 0

    def build(self, texts: List[str]):
        """
        Строим матрицу term-document.
        """

        all_terms = set()
        for text in texts:
            tokens = re.findall(r"\w+|[^\w\s]", text)
            all_terms.update(tokens)
        self.term_to_idx = {
            term: i for i, term in enumerate(
                sorted(all_terms))}

        self.num_docs = len(texts)
        num_terms = len(self.term_to_idx)
        self.matrix = lil_matrix((num_terms, self.num_docs), dtype=int)

        for doc_id, text in enumerate(texts):
            tokens = re.findall(r"\w+|[^\w\s]", text)
            freq = Counter(tokens)
            for token, count in freq.items():
                row = self.term_to_idx[token]
                self.matrix[row, doc_id] = count

    def save(self, path: str):
        """
        Сохраняем матрицу и словарь терминов на диск.
        """
        # Сохраняем матрицу в npz
        matrix_csr = self.matrix.tocsr()
        save_npz(path + ".npz", matrix_csr)

        # Сохраняем словарь и количество документов
        with open(path + "_meta.pkl", "wb") as f:
            pickle.dump({
                "term_to_idx": self.term_to_idx,
                "num_docs": self.num_docs
            }, f)

    def load(self, path: str):
        """
        Загружаем матрицу и словарь терминов с диска.
        """
        # Загружаем матрицу
        self.matrix = load_npz(path + ".npz")

        # Загружаем словарь и количество документов
        with open(path + "_meta.pkl", "rb") as f:
            data = pickle.load(f)
            self.term_to_idx = data["term_to_idx"]
            self.num_docs = data["num_docs"]

    def get_matrix(self):
        """Возвращаем саму разреженную матрицу."""
        return self.matrix

    def get_terms(self):
        """Возвращаем словарь термин -> индекс строки."""
        return self.term_to_idx

    def get_num_docs(self):
        """Возвращаем количество документов."""
        return self.num_docs

    def search(self, query: str, top_k: int = 5):
        tokens = re.findall(r"\w+|[^\w\s]", query)
        scores = {}

        matrix = self.matrix.tocsr()

        for token in tokens:
            if token in self.term_to_idx:
                row = self.term_to_idx[token]
                row_data = matrix.getrow(row)

                for doc_id, freq in zip(row_data.indices, row_data.data):
                    scores[doc_id] = scores.get(doc_id, 0) + freq

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
