class BaseIndex():
    """
    Базовый класс для всех индексов.
    Определяет интерфейс для построения, сохранения, загрузки и поиска.
    """

    def build(self, texts):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def search(self, query: str, top_k: int = 5):
        pass
