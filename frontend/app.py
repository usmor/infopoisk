from flask import Flask, render_template, request, jsonify, make_response
import os
import sys
from translations import TRANSLATIONS

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.src.search import SearchEngine # noqa: E402

app = Flask(__name__)

search_engine = SearchEngine()


@app.route('/')
def index():
    """
    Главная страница
    """
    lang = request.args.get('lang', request.cookies.get('lang', 'ru'))

    if lang not in TRANSLATIONS:
        lang = 'ru'

    translations = TRANSLATIONS[lang].copy()
    translations['lang'] = lang

    response = make_response(render_template('index.html', **translations))

    response.set_cookie('lang', lang, max_age=365*24*60*60)

    return response


@app.route('/api/translate')
def api_translate():
    lang = request.args.get('lang', 'ru')

    if lang not in TRANSLATIONS:
        lang = 'ru'

    return jsonify(TRANSLATIONS[lang])


@app.route('/search')
def search_page():
    """
    Страница поиска
    """
    lang = request.args.get('lang', request.cookies.get('lang', 'ru'))

    if lang not in TRANSLATIONS:
        lang = 'ru'

    translations = TRANSLATIONS[lang].copy()
    translations['lang'] = lang

    return render_template('search.html', **translations)


@app.route('/results')
def results_page():
    """
    Страница результатов
    """
    query = request.args.get('query', '').strip()
    index_type = request.args.get('index-switch', 'bm25')
    top_k = request.args.get('k_top', 10, type=int)

    lang = request.args.get('lang', request.cookies.get('lang', 'ru'))

    if lang not in TRANSLATIONS:
        lang = 'ru'

    translations = TRANSLATIONS[lang].copy()
    translations['lang'] = lang

    if not query:
        return render_template('search.html',
                               error="Пожалуйста, введите поисковый запрос",
                               **translations)

    try:
        results = search_engine.search(query, index_type, top_k)

        return render_template('results.html',
                               query=query,
                               results=results,
                               index_type=index_type,
                               top_k=top_k,
                               search_time=results.get('search_time', 0),
                               total_results=len(results.get('results', [])),
                               **translations)
    except Exception as e:
        return render_template('results.html',
                               query=query,
                               error=str(e),
                               index_type=index_type,
                               top_k=top_k,
                               **translations)
    
@app.route('/api/talk_translation')
def api_talk_translation():
    """
    API для получения перевода выступления на указанный язык
    """
    talk_id = request.args.get('talk_id', type=int)
    lang = request.args.get('lang', 'ru')
    
    if not talk_id:
        return jsonify({'error': 'No talk_id provided'}), 400
    
    try:
        title = search_engine.storage.get_title(talk_id, lang)
        description = search_engine.storage.get_description(talk_id, lang)
        transcript = search_engine.storage.get_transcript(talk_id, lang)
    
        labels = TRANSLATIONS.get(lang, TRANSLATIONS['ru'])

        return jsonify({
            'title': title,
            'description': description,
            'transcript': transcript,
            'language': lang,
            'talk_id': talk_id,
            'labels': {
                'description_label': labels.get('description', 'Description'),
                'transcript_label': labels.get('transcript_label', 'Transcript')
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
