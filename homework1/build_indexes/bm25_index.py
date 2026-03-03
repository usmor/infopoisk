import bm25s
import pickle
import re
import numpy as np
from typing import List
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix, save_npz, load_npz
import sys
import os
from rank_bm25 import BM25Okapi
sys.path.append(os.path.dirname(__file__))
from base_index import BaseIndex  # noqa: E402


# class BM25IndexLib(BaseIndex):
#     """
#     Обратный индекс BM25 строится через библиотекуbm25s.
#     """

#     def __init__(self, **kwargs):
#         """
#         Можно передать параметры BM25 (например method, k1, b).
#         """
#         self.retriever = None
#         self.kwargs = kwargs
#         self.corpus_tokens = []

#     def build(self, texts: List[str]):
#         self.corpus_tokens = bm25s.tokenize(texts)
#         self.retriever = bm25s.BM25(**self.kwargs)
#         self.retriever.index(self.corpus_tokens)

#     def save(self, path: str):
#         with open(path, "wb") as f:
#             pickle.dump((self.corpus_tokens, self.kwargs), f)

#     def load(self, path: str):
#         with open(path, "rb") as f:
#             self.corpus_tokens, self.kwargs = pickle.load(f)
#         self.retriever = bm25s.BM25(**self.kwargs)
#         self.retriever.index(self.corpus_tokens)

#     def search(self, query: str, top_k: int = 5):
#         query_tokens = bm25s.tokenize([query])[0][0]
#         scores = self.retriever.get_scores(query_tokens)

#         top_ids = np.argsort(scores)[::-1][:top_k]

#         return [(int(doc_id), float(scores[doc_id])) for doc_id in top_ids]

class BM25IndexLib(BaseIndex):
    """
    Обратный индекс BM25 строится через библиотеку rank-bm25.
    """

    def __init__(self, k1=1.5, b=0.75, **kwargs):
        self.bm25_type = "okapi"
        self.k1 = k1
        self.b = b
        self.kwargs = kwargs
        self.retriever = None
        self.corpus = None
        self.tokenized_corpus = None

    def build(self, texts: List[str]):
        """Построение BM25 индекса."""
        self.corpus = texts
        self.tokenized_corpus = [re.findall(r"\w+|[^\w\s]", text) for text in texts]
        self.retriever = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b, **self.kwargs)

    def save(self, path: str):
        """Сохраняем индекс и параметры."""
        with open(path, "wb") as f:
            pickle.dump({
                'corpus': self.corpus,
                'tokenized_corpus': self.tokenized_corpus,
                'bm25_type': self.bm25_type,
                'k1': self.k1,
                'b': self.b,
                'kwargs': self.kwargs
            }, f)

    def load(self, path: str):
        """Загружаем индекс и параметры."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        self.corpus = data['corpus']
        self.tokenized_corpus = data['tokenized_corpus']
        self.k1 = data['k1']
        self.b = data['b']
        self.kwargs = data['kwargs']
        
        self.build(self.corpus)

    def search(self, query: str, top_k: int = 5):
        """Поиск по запросу."""
        if self.retriever is None:
            raise ValueError("Индекс не построен. Сначала вызовите build()")
        
        tokenized_query = re.findall(r"\w+|[^\w\s]", query)
        
        scores = self.retriever.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(i), float(scores[i])) for i in top_indices]

class BM25IndexDict(BaseIndex):
    """
    BM25 с обратным индексом в виде словаря.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.inverted_index = {}
        self.doc_lens = {}
        self.avg_dl = 0
        self.num_docs = 0
        self.k1 = k1
        self.b = b

    def build(self, texts: List[str]):
        """Строим обратный индекс в виде словаря."""
        self.num_docs = len(texts)
        inverted_index = defaultdict(dict)
        doc_lens = {}

        for doc_id, text in enumerate(texts):
            tokens = re.findall(r"\w+|[^\w\s]", text)
            freq = Counter(tokens)
            doc_lens[doc_id] = len(tokens)
            for term, f in freq.items():
                inverted_index[term][doc_id] = f

        self.inverted_index = dict(inverted_index)
        self.doc_lens = doc_lens
        self.avg_dl = sum(doc_lens.values()) / self.num_docs

    def save(self, path: str):
        """Сохраняем словарь и метаданные."""
        with open(path, "wb") as f:
            pickle.dump(
                (self.inverted_index,
                 self.doc_lens,
                 self.avg_dl,
                 self.num_docs,
                 self.k1,
                 self.b),
                f)

    def load(self, path: str):
        """Загружаем словарь и метаданные."""
        (self.inverted_index, self.doc_lens,
         self.avg_dl, self.num_docs,
         self.k1, self.b) = pickle.load(open(path, "rb"))

    def search(self, query: str, top_k: int = 5):
        tokens = re.findall(r"\w+|[^\w\s]", query)
        scores = {}

        N = self.num_docs
        avg_dl = self.avg_dl
        k1 = self.k1
        b = self.b

        for token in tokens:
            if token not in self.inverted_index:
                continue

            posting = self.inverted_index[token]
            df = len(posting)

            idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in posting.items():
                dl = self.doc_lens[doc_id]

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avg_dl)

                score = idf * numerator / denominator

                scores[doc_id] = scores.get(doc_id, 0) + score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class BM25IndexMatrix(BaseIndex):
    """
    BM25 с разреженной матрицей term-document.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.term_to_idx = {}
        self.matrix = None
        self.doc_lens = None
        self.avg_dl = 0
        self.num_docs = 0
        self.k1 = k1
        self.b = b

    def build(self, texts: List[str]):
        """Строим разреженную матрицу term-document."""
        tokenized = []
        all_terms = set()
        self.num_docs = len(texts)

        for text in texts:
            tokens = re.findall(r"\w+|[^\w\s]", text)
            tokenized.append(tokens)
            all_terms.update(tokens)

        self.doc_lens = np.array([len(toks) for toks in tokenized])
        self.avg_dl = np.mean(self.doc_lens)

        self.term_to_idx = {
            term: i for i, term in enumerate(
                sorted(all_terms))}
        self.matrix = lil_matrix(
            (len(self.term_to_idx), self.num_docs), dtype=int)

        for doc_id, tokens in enumerate(tokenized):
            freq = Counter(tokens)
            for term, f in freq.items():
                self.matrix[self.term_to_idx[term], doc_id] = f

    def save(self, path: str):
        """Сохраняем разреженную матрицу и метаданные."""
        matrix_csr = self.matrix.tocsr()
        save_npz(path + ".npz", matrix_csr)
        with open(path + "_meta.pkl", "wb") as f:
            pickle.dump((self.term_to_idx, self.doc_lens,
                         self.avg_dl, self.num_docs, self.k1, self.b), f)

    def load(self, path: str):
        """Загружаем матрицу и метаданные."""
        self.matrix = load_npz(path + ".npz")
        with open(path + "_meta.pkl", "rb") as f:
            (self.term_to_idx, self.doc_lens,
             self.avg_dl, self.num_docs,
             self.k1, self.b) = pickle.load(f)

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
        N = self.num_docs
        avg_dl = self.avg_dl
        k1 = self.k1
        b = self.b

        for token in tokens:
            if token not in self.term_to_idx:
                continue

            row = self.term_to_idx[token]
            row_data = matrix.getrow(row)

            df = len(row_data.indices)
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in zip(row_data.indices, row_data.data):
                dl = self.doc_lens[doc_id]

                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avg_dl)

                score = idf * numerator / denominator

                scores[doc_id] = scores.get(doc_id, 0) + score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
