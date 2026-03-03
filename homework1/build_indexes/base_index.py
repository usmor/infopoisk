from abc import ABC, abstractmethod


class BaseIndex(ABC):
    """
    Базовый класс для всех индексов.
    Определяет интерфейс для построения, сохранения и загрузки.
    """

    @abstractmethod
    def build(self, texts):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5):
        pass
