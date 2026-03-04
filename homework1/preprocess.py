import pymorphy3
from razdel import tokenize
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tqdm.auto import tqdm
nltk.download('stopwords')

# Инициализация морфологического анализатора
morph = pymorphy3.MorphAnalyzer()

# Загружаем стоп-слова для русского языка
nltk_stopwords = set(stopwords.words('russian'))


# Функция препроцессинга текста
def preprocess_text(text: str) -> str:
    """
      Предобработка текста:
      - очистка от пунктуации и чисел
      - лемматизация
      - удаление стоп-слов
    """

    if not isinstance(text, str) or pd.isna(text):
        return ""

    # Обрабатываем текст с помощью Razdel и Pymorphy3
    tokens = [t.text.lower() for t in tokenize(text)]
    lemmas = [
        morph.parse(token)[0].normal_form
        for token in tokens
        if token.isalpha() and token not in nltk_stopwords and len(token) > 2
    ]

    return ' '.join(lemmas)


# Функция для обработки всего корпуса рецептов
def preprocess_corpus() -> list:
    df = pd.read_csv(
        "./tales.csv",
        sep=",",
        quotechar='"',
        on_bad_lines='skip',
        encoding="utf-8"
    )

    # Выбираем только детские сказки
    df = df[df["Label"] == 1].reset_index(drop=True)

    # Применяем препроцессинг к каждому тексту и отслеживаем прогресс
    tqdm.pandas()
    df["preprocessed_tale"] = df["Tale"].progress_apply(preprocess_text)

    # Удаляем столбец с меткой, он больше не нужен
    del df["Label"]

    # Сохраняем предобработанные тексты в новый CSV файл
    df.to_csv("./preprocessed_tales.csv", index=False)

    # Данный список текстов будет использоваться для построения индексов
    return df["preprocessed_tale"].tolist()
