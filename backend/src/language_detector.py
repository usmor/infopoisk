from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42


def language_detector(text: str, default: str = 'en') -> str:
    """
    Определение языка текста запроса.
    Поддерживает основные языки: русский, английский, французский.
    """
    SUPPORTED_LANGUAGES = {'ru', 'en', 'fr'}

    if not text:
        return default

    try:
        detected = detect_langs(text)
        for lang in detected:
            if lang.lang in SUPPORTED_LANGUAGES:
                return lang.lang
        return default

    except LangDetectException:
        return default
