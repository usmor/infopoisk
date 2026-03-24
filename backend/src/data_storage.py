import json
import os
from typing import Dict, List, Optional, Any

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FULL_DATA_PATH = os.path.join(DATA_DIR, "ted_documents.json")


class DataStorage:
    """
    Хранилище данных TED Talks из ted_documents.json.
    Предназначено для удобного доступа к текстам,
    заголовкам, описаниям и метаданным.
    """

    def __init__(self):
        self.docs_by_id = {}          # talk_id -> весь документ
        self.transcripts_by_id = {}    # talk_id -> {lang: transcript}
        self.titles_by_id = {}          # talk_id -> {lang: title}
        self.descriptions_by_id = {}    # talk_id -> {lang: description}
        self.load_data()

    def load_data(self) -> None:
        """Загружает данные из ted_documents.json."""
        if not os.path.exists(FULL_DATA_PATH):
            raise FileNotFoundError(
                f"Файл не найден: {FULL_DATA_PATH}\n"
            )

        with open(FULL_DATA_PATH, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        for doc in raw_docs:
            talk_id = doc['talk_id']
            self.docs_by_id[talk_id] = doc
            self.transcripts_by_id[talk_id] = doc.get('transcripts', {})
            self.titles_by_id[talk_id] = doc.get('title', {})
            self.descriptions_by_id[talk_id] = doc.get('descriptions', {})

    def get_transcript(self, talk_id: int, language: str) -> Optional[str]:
        """
        Возвращает транскрипт на указанном языке.
        """
        return self.transcripts_by_id.get(talk_id, {}).get(language)

    def get_available_languages(self, talk_id: int) -> List[str]:
        """
        Возвращает список доступных языков для выступления.
        """
        return list(self.transcripts_by_id.get(talk_id, {}).keys())

    def get_title(self, talk_id: int, language: str = "ru") -> str:
        """
        Возвращает название выступления на указанном языке.
        """
        titles = self.titles_by_id.get(talk_id, {})

        if language in titles:
            return titles[language]

        return f"Выступление #{talk_id}"

    def get_description(self, talk_id: int,
                        language: str = "ru") -> Optional[str]:
        """
        Возвращает описание выступления на указанном языке.
        """
        descriptions = self.descriptions_by_id.get(talk_id, {})
        return descriptions.get(language) or descriptions.get('en')

    def get_speakers(self, talk_id: int) -> List[str]:
        """
        Возвращает список спикеров.
        """
        doc = self.docs_by_id.get(talk_id, {})
        return doc.get('speakers', [])

    def get_url(self, talk_id: int) -> str:
        """
        Возвращает URL выступления.
        """
        doc = self.docs_by_id.get(talk_id, {})
        return doc.get('url', '')

    def get_all_metadata(self, talk_id: int) -> Dict[str, Any]:
        """
        Возвращает все метаданные выступления.
        """
        return self.docs_by_id.get(talk_id, {})
