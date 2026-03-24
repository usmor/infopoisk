import argparse
from search import SearchEngine


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Messages:
    NO_RESULTS = {
        'en': "🔍 No results found for '{}'",
        'fr': "🔍 Aucun résultat trouvé pour '{}'",
        'ru': "🔍 Результаты не найдены для '{}'"
    }
    RELEVANCE = {
        'en': "📊 Relevance: {}",
        'fr': "📊 Rélevance: {}",
        'ru': "📊 Релевантность: {}"
    }
    QUERY_INFO = {
        'en': "🔍 QUERY: '{}'",
        'fr': "🔍 REQUÊTE: '{}'",
        'ru': "🔍 ЗАПРОС: '{}'"
    }
    QUERY_LANG = {
        'en': "🌐 QUERY LANGUAGE: {}",
        'fr': "🌐 LANGUE DE LA REQUÊTE: {}",
        'ru': "🌐 ЯЗЫК ЗАПРОСА: {}"
    }
    SEARCH_TIME = {
        'en': "⌛️ SEARCH TIME: {} sec",
        'fr': "⌛️ TEMPS DE RECHERCHE: {} sec",
        'ru': "⌛️ ВРЕМЯ ПОИСКА: {} сек"
    }
    MODEL_USED = {
        'en': "📊 MODEL: {}",
        'fr': "📊 MODÈLE: {}",
        'ru': "📊 МОДЕЛЬ: {}"
    }
    OTHER_LANGS = {
        'en': "🌍 Available in other languages: {}",
        'fr': "🌍 Disponible dans d'autres langues: {}",
        'ru': "🌍 Доступно на других языках: {}"
    }
    TOTAL_RESULTS = {
        'en': "📑 TOTAL RESULTS: {}",
        'fr': "📑 RÉSULTATS TOTAUX: {}",
        'ru': "📑 ВСЕГО РЕЗУЛЬТАТОВ: {}"
    }
    TOTAL_TIME = {
        'en': "⏰ TOTAL TIME: {} sec",
        'fr': "⏰ TEMPS TOTAL: {} sec",
        'ru': "⏰ ПОЛНОЕ ВРЕМЯ: {} сек"
    }
    URL = {
        'en': "🔗 LINK TO VIDEO: {}",
        'fr': "🔗 URL: {}",
        'ru': "🔗 ССЫЛКА НА ВИДЕО: {}"
    }


engine = SearchEngine()


def format_result(result, query_lang):
    """
    Форматирует один результат.
    """
    score = result['score']
    lang = result['language']

    title = result['title'].strip()
    speakers = result['speakers']
    url = result['url']
    desc = result['description'].strip()
    available = result['available_languages']

    lang_emojis = {'en': '🇬🇧', 'fr': '🇫🇷', 'ru': '🇷🇺'}

    other_langs = []
    for av_lang in available:
        if av_lang != lang:
            emoji = lang_emojis.get(av_lang, av_lang)
            other_langs.append(f"{emoji} {av_lang}")

    title = f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}"
    score_str = f"{Colors.GREEN}{score:.3f}{Colors.END}"

    lines = []
    lines.append(f"📌 {title}")
    lines.append(f"   🎤 {', '.join(speakers).strip()}")
    lines.append(f"   🔗 URL {url}")
    lines.append(f"   {Messages.RELEVANCE[query_lang].format(score_str)}")

    if desc:
        lines.append(f"   📋 {desc}")

    if other_langs:
        other_langs_str = ', '.join(other_langs)
        template = Messages.OTHER_LANGS[query_lang]
        formatted_msg = template.format(other_langs_str)
        lines.append(f"   {formatted_msg}")

    return "\n".join(lines)


def format_all_results(results_dict):
    """
    Форматирует все результаты.
    """
    lines = []
    lines.append("\n" + "=" * 80)
    if len(results_dict['results']) == 0:
        lines.append(Messages.NO_RESULTS[results_dict['query_lang']].format(
            results_dict['query']))
        return "\n".join(lines)

    lines.append(Messages.QUERY_INFO[results_dict['query_lang']].format(
        results_dict['query']))
    lines.append(Messages.QUERY_LANG[results_dict['query_lang']].format(
        results_dict['query_lang']))
    lines.append(Messages.MODEL_USED[results_dict['query_lang']].format(
        results_dict['index_type']))
    lines.append(Messages.SEARCH_TIME[results_dict['query_lang']].format(
        results_dict['search_time']))

    lines.append(Messages.TOTAL_RESULTS[results_dict['query_lang']].format(
        len(results_dict['results'])))
    lines.append("-" * 80)

    for i, result in enumerate(results_dict['results'], 1):
        formatted = format_result(result, results_dict['query_lang'])
        lines.append(f"\n{i}. {formatted}")
        lines.append("-" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='🔍 Поиск по TED Talks на трёх языках',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python cli.py --query "искусственный интеллект"
  python cli.py -q "climate change" -i word2vec -k 5
  python cli.py --query "amour" --index fasttext --top-k 3
        """
    )
    parser.add_argument('--query', '-q', required=True, help='Текст запроса')
    parser.add_argument('--index', '-i', default='bm25',
                        choices=['bm25', 'word2vec', 'fasttext'],
                        help='Тип индекса (bm25/word2vec/fasttext)')
    parser.add_argument('--top-k', '-k', type=int, default=10,
                        help='Количество результатов')

    args = parser.parse_args()

    print(f"🚀 Поиск: '{args.query}' [индекс: {args.index}]...")

    try:
        results_dict = engine.search(args.query, args.index, args.top_k)
        formatted = format_all_results(results_dict)
        print(formatted)
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
