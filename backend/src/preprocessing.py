import json
import re
import spacy
import pymorphy3
import os
from razdel import tokenize
import nltk
from nltk.corpus import stopwords
from tqdm.auto import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "ted_documents.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "ted_preprocessed.json")

try:
    stop_words_en = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    stop_words_ru = set(stopwords.words('russian'))
except LookupError:
    print("Стоп-слова не найдены. Скачивание...")
    nltk.download('stopwords', quiet=True)
    stop_words_en = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    stop_words_ru = set(stopwords.words('russian'))

nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")
morph_ru = pymorphy3.MorphAnalyzer()


def clean_text(text: str) -> str:
    """Базовая очистка текста
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_for_index(text: str, lang: str):
    """
    Лемматизация для построения индексов.
    """
    text = clean_text(text)

    if lang == "en":
        doc = nlp_en(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.text.lower() not in stop_words_en
        ]
    elif lang == "fr":
        doc = nlp_fr(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.text.lower() not in stop_words_fr
        ]
    elif lang == "ru":
        tokens = [token.text.lower()
                  for token in tokenize(text) if token.text.isalpha()]
        return [
            morph_ru.parse(token)[0].normal_form
            for token in tokens
            if token not in stop_words_ru
        ]
    else:
        return text.split()


def lemmatize_for_query(text: str, lang: str):
    """
    Лемматизация для поискового запроса.
    """
    text = clean_text(text)

    if lang == "en":
        doc = nlp_en(text)
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        return lemmas

    elif lang == "fr":
        doc = nlp_fr(text)
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        return lemmas

    elif lang == "ru":
        tokens = [token.text.lower()
                  for token in tokenize(text) if token.text.isalpha()]
        lemmas = [morph_ru.parse(token)[0].normal_form for token in tokens]
        return lemmas

    else:
        return text.split()


def preprocess_corpus():
    """Обработка всего корпуса"""
    print(f"Загрузка данных из {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    preprocessed_docs = []
    for doc in tqdm(documents, desc="Обработка документов"):
        lemmatized = {}
        for lang, transcript in doc["transcripts"].items():
            lemmas = lemmatize_for_index(transcript, lang)
            lemmatized[lang] = ' '.join(lemmas)

        preprocessed_docs.append({
            "talk_id": doc["talk_id"],
            "lemmatized_transcripts": lemmatized
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(preprocessed_docs, f, ensure_ascii=False, indent=2)

    print(f"Обработанный корпус сохранен: {OUTPUT_PATH}")
    return preprocessed_docs


if __name__ == "__main__":
    preprocess_corpus()
