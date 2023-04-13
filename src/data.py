import json
import pandas as pd


def jsonl_to_df(data_json: list) -> pd.DataFrame:
    return pd.DataFrame(
        [{'id': i['uuid'],
          'title': i['targetTitle'],
          'question': ' '.join(i['postText']),
          'context': i['targetTitle'] + ' - ' +
          (' '.join(i['targetParagraphs'])),
          'spoiler': i['spoiler']} for i in data_json])


def read_data(filename: str) -> pd.DataFrame:
    open_json = [json.loads(i) for i in open(filename, 'r')]
    dataset = jsonl_to_df(open_json)
    return dataset
