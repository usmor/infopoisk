import pandas as pd
import os
import json
import ast


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

EN_PATH = os.path.join(DATA_DIR, "ted_talks_en.csv")
FR_PATH = os.path.join(DATA_DIR, "ted_talks_fr.csv")
RU_PATH = os.path.join(DATA_DIR, "ted_talks_ru.csv")

en_df = pd.read_csv(EN_PATH)
fr_df = pd.read_csv(FR_PATH)
ru_df = pd.read_csv(RU_PATH)

en_map = en_df.set_index('talk_id').to_dict('index')
fr_map = fr_df.set_index('talk_id').to_dict('index')
ru_map = ru_df.set_index('talk_id').to_dict('index')

en_ids = set(en_df['talk_id'])
fr_ids = set(fr_df['talk_id'])
ru_ids = set(ru_df['talk_id'])
all_ids = en_ids.union(fr_ids).union(ru_ids)

documents = []

for talk_id in all_ids:
    en = en_map.get(talk_id)
    fr = fr_map.get(talk_id)
    ru = ru_map.get(talk_id)

    meta_source = en or fr or ru
    if not meta_source:
        continue

    transcripts = {}
    descriptions = {}
    titles = {}

    if en and pd.notna(en.get('transcript')):
        transcripts['en'] = en.get('transcript')
        descriptions['en'] = en.get('description')
        titles['en'] = en.get('title')

    if fr and pd.notna(fr.get('transcript')):
        transcripts['fr'] = fr.get('transcript')
        descriptions['fr'] = fr.get('description')
        titles['fr'] = fr.get('title')

    if ru and pd.notna(ru.get('transcript')):
        transcripts['ru'] = ru.get('transcript')
        descriptions['ru'] = ru.get('description')
        titles['ru'] = ru.get('title')

    speakers = meta_source.get('all_speakers')
    if speakers:
        try:
            speakers_dict = ast.literal_eval(speakers)
            speakers_list = list(speakers_dict.values())
        except BaseException:
            speakers_list = []
    else:
        speakers_list = []

    documents.append({
        'talk_id': talk_id,
        'title': titles,
        'url': meta_source.get('url'),
        'speakers': speakers_list,
        'transcripts': transcripts,
        'descriptions': descriptions
    })

OUTPUT_PATH = os.path.join(DATA_DIR, "ted_documents.json")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"Собрано {len(documents)} документов. Сохранено в {OUTPUT_PATH}")
