import pickle
import pandas as pd

df = pd.read_csv("preprocessed_tales.csv")
with open("./builded_indexes/index_paths.pkl", "rb") as f:
    index_paths = pickle.load(f)


def search():
    """
    Универсальная функция поиска по любому индексу.
    """

    # Получаем параметры от пользователя
    query = input("Введите поисковый запрос: ").strip()
    index_type = input("Выберите тип индекса (freq или bm25): ").strip()
    sub_type = input(
        "Выберите подтип индекса (lib, dict или matrix): ").strip()
    top_k = int(input("Введите количество результатов для отображения: "))

    print("\nИщем...\n")

    key = f"{index_type}_{sub_type}"
    if key not in index_paths:
        raise ValueError(f"Индекс {key} не найден")
    path = index_paths[key]

    # Динамический импорт нужного класса
    if index_type == "freq":
        if sub_type == "lib":
            from build_indexes.freq_index import (
                FrequencyIndexLib as IndexClass
            )
        elif sub_type == "dict":
            from build_indexes.freq_index import (
                FrequencyIndexDict as IndexClass
            )
        elif sub_type == "matrix":
            from build_indexes.freq_index import (
                FrequencyIndexMatrix as IndexClass
            )
    elif index_type == "bm25":
        if sub_type == "lib":
            from build_indexes.bm25_index import BM25IndexLib as IndexClass
        elif sub_type == "dict":
            from build_indexes.bm25_index import BM25IndexDict as IndexClass
        elif sub_type == "matrix":
            from build_indexes.bm25_index import BM25IndexMatrix as IndexClass
    else:
        raise ValueError(f"Неизвестный тип индекса: {index_type}")

    # Создаём индекс
    if index_type == "freq" and sub_type == "lib":
        # Для Whoosh индекса передаём директорию
        idx = IndexClass(index_dir=path)
    else:
        idx = IndexClass()
        idx.load(path)

    # Поиск
    results = idx.search(query, top_k=top_k)
    results = [(doc_id, score) for doc_id, score in results if score > 0]

    if len(results) == 0:
        print("Ничего не найдено.")
        return []

    print(f"Всего документов найдено: {len(results)}.")

    # Формируем вывод
    output = []
    for doc_id, score in results:
        if doc_id >= len(df):
            continue
        row = df.iloc[doc_id]
        snippet = (row["Tale"][:1000] +
                   "...") if len(row["Tale"]) > 1000 else row["Tale"]
        output.append({
            "snippet": snippet,
            "score": score
        })

    # Красивый вывод
    for i, item in enumerate(output, 1):
        print(f"🔹  Документ № {i}.  (score: {item['score']:.2f})")
        print(f"    {item['snippet']}\n")

    return output
