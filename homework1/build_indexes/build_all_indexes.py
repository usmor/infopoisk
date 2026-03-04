from bm25_index import BM25IndexLib, BM25IndexDict, BM25IndexMatrix
from freq_index import (
    FrequencyIndexDict,
    FrequencyIndexMatrix,
    FrequencyIndexLib,
)
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocess import preprocess_corpus  # noqa: E402


def build_all_indexes(
        save_dir: str = "builded_indexes",
        paths_file: str = "index_paths.pkl"):
    """
    Строит все индексы (freq и BM25, lib/dict/matrix) и сохраняет их.
    Сохраняет пути к индексам в один файл paths_file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Перед построением индексов корпус текстов проходит обработку
    print("Шаг 1. Препроцессинг корпуса...")
    texts = preprocess_corpus()
    print(f"Корпус обработан. Всего документов: {len(texts)}")

    # Словарь для хранения путей к каждому индексу
    index_paths = {}

    # Последовательное построение всех индексов
    print("Шаг 2. Строим частотные индексы...")
    # FrequencyIndexLib: для Whoosh индекса передаём директорию, так как он сохраняет несколько файлов
    freq_lib_index = FrequencyIndexLib(
        index_dir=os.path.join(
            save_dir, "freq_index_lib"))
    freq_lib_index.build(texts)
    index_paths["freq_lib"] = os.path.join(save_dir, "freq_index_lib")

    # FrequencyIndexDict: сохраняем в один файл .pkl, так как это словарь
    freq_dict_index = FrequencyIndexDict()
    freq_dict_index.build(texts)
    dict_path = os.path.join(save_dir, "freq_index_dict.pkl")
    freq_dict_index.save(dict_path)
    index_paths["freq_dict"] = dict_path

    # FrequencyIndexMatrix: сохраняются и матрица, и метаданные
    freq_matrix_index = FrequencyIndexMatrix()
    freq_matrix_index.build(texts)
    matrix_path = os.path.join(save_dir, "freq_index_matrix")
    freq_matrix_index.save(matrix_path)
    index_paths["freq_matrix"] = matrix_path

    print("Шаг 3. Строим BM25 индексы...")
    # BM25IndexLib: библиотека rank-bm25 сохраняет только словарь, поэтому сохраняем его в .pkl файл
    bm25_lib_index = BM25IndexLib()
    bm25_lib_index.build(texts)
    lib_path = os.path.join(save_dir, "bm25_lib.pkl")
    bm25_lib_index.save(lib_path)
    index_paths["bm25_lib"] = lib_path

    # BM25IndexDict: сохраняем в один файл .pkl, так как это словарь
    bm25_dict_index = BM25IndexDict()
    bm25_dict_index.build(texts)
    dict_path = os.path.join(save_dir, "bm25_dict.pkl")
    bm25_dict_index.save(dict_path)
    index_paths["bm25_dict"] = dict_path

    # BM25IndexMatrix: сохраняются и матрица, и метаданные
    bm25_matrix_index = BM25IndexMatrix()
    bm25_matrix_index.build(texts)
    matrix_path = os.path.join(save_dir, "bm25_matrix")
    bm25_matrix_index.save(matrix_path)
    index_paths["bm25_matrix"] = matrix_path

    # Сохраняем пути в один файл
    paths_file_path = os.path.join(save_dir, paths_file)
    with open(paths_file_path, "wb") as f:
        pickle.dump(index_paths, f)

    print("Все индексы созданы и пути сохранены в:", paths_file_path)
    return index_paths
