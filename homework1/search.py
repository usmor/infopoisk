import pickle
from typing import Optional
import pandas as pd

# Загрузка предобработанных текстов и путей к индексам
df = pd.read_csv("preprocessed_tales.csv")
with open("./builded_indexes/index_paths.pkl", "rb") as f:
    index_paths = pickle.load(f)

def search(snippet_len: Optional[int] = None):
    """
    Универсальная функция поиска по любому индексу.
    snippet_len: optional
        Если указан, ограничивает длину выводимого текста.
    """

    try:
        # Получаем параметры от пользователя
        query = input("Введите поисковый запрос: ").strip()
        if not query:
            raise ValueError("Поисковый запрос не может быть пустым")

        index_type = input(
            "Выберите тип индекса (freq или bm25): ").strip().lower()
        if not index_type:
            raise ValueError("Тип индекса не может быть пустым")
        if index_type not in ["freq", "bm25"]:
            raise ValueError("Тип индекса должен быть 'freq' или 'bm25'")
        
        sub_type = input(
            "Выберите подтип индекса (lib, dict или matrix): ").strip().lower()
        if not sub_type:
            raise ValueError("Подтип индекса не может быть пустым")
        if sub_type not in ["lib", "dict", "matrix"]:
            raise ValueError("Подтип индекса должен быть 'lib', 'dict' или 'matrix'")

        try:
            top_k = int(
                input("Введите количество результатов для отображения: "))
        except ValueError:
            raise ValueError("Количество результатов должно быть числом")

        print("\nИщем...\n")

        # Проверяем наличие индекса
        key = f"{index_type}_{sub_type}"
        if key not in index_paths:
            raise ValueError(f"Индекс {key} не найден")

        path = index_paths[key]

        # Динамический импорт нужного класса
        if index_type == "freq":
            if sub_type == "lib":
                from build_indexes.freq_index import FrequencyIndexLib as IndexClass
            elif sub_type == "dict":
                from build_indexes.freq_index import FrequencyIndexDict as IndexClass
            elif sub_type == "matrix":
                from build_indexes.freq_index import FrequencyIndexMatrix as IndexClass
            else:
                raise ValueError("Неизвестный подтип индекса")
        elif index_type == "bm25":
            if sub_type == "lib":
                from build_indexes.bm25_index import BM25IndexLib as IndexClass
            elif sub_type == "dict":
                from build_indexes.bm25_index import BM25IndexDict as IndexClass
            elif sub_type == "matrix":
                from build_indexes.bm25_index import BM25IndexMatrix as IndexClass
            else:
                raise ValueError("Неизвестный подтип индекса")
        else:
            raise ValueError(f"Неизвестный тип индекса: {index_type}")

        # Создаём индекс
        if index_type == "freq" and sub_type == "lib":
            idx = IndexClass(index_dir=path)
        else:
            idx = IndexClass()
            idx.load(path)

        # Поиск
        results = idx.search(query, top_k=top_k)

        # Фильтруем нулевые результаты
        results = [(doc_id, score) for doc_id, score in results if score > 0]

        if len(results) == 0:
            print("Ничего не найдено.")
            return []

        print(f"Всего документов найдено: {len(results)}.")

        output = []

        for doc_id, score in results:
            row = df.iloc[doc_id]
            text = row["Tale"]

            # Cоздаём snippet если указан соответсвующий параметр
            if snippet_len is not None and len(text) > snippet_len:
                snippet = text[:snippet_len] + "..."
            else:
                snippet = text

            output.append({
                "snippet": snippet,
                "score": score
            })

        # Красивый вывод
        for i, item in enumerate(output, 1):
            print(f"🔹  Документ № {i}.  (score: {item['score']:.2f})")
            print(f"    {item['snippet']}\n")

        return output

    except Exception as e:
        print(f"Ошибка: {e}")
        return []
