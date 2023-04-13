import json
import pandas as pd


def read_data(filename: str) -> pd.DataFrame:
    """
    Read data into dataframe from provided filename

    :param filename: name of the file to load
    :return: dataframe with contents of the provided file
    """
    data_json = [json.loads(i) for i in open(filename, 'r')]
    df = pd.DataFrame(
        [{'id': i['uuid'],
          'title': i['targetTitle'],
          'question': ' '.join(i['postText']),
          'context': i['targetTitle'] + ' - ' +
          (' '.join(i['targetParagraphs'])),
          'spoiler': i['spoiler']} for i in data_json])
    return df
