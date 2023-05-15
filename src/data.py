import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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


def get_encodings(df: pd.DataFrame):
    ...


class ClickbaitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer_model: str) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.encodings = self.tokenizer(
            self.df.context, self.df.question, truncation=True, padding=True)
        self.encodings.update({'encoding_ids': np.arange(len(df)),
                               "contexts": df.context.tolist(),
                               "questions": df.question.tolist()})
        # TODO: add spoiler positions

    def __len__(self) -> int:
        return len(self.encodings.input_ids)

    def __getitem__(self, idx: int) -> dict:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        batch = {}
        for key, val in self.encodings.items():
            if type(val[idx]) == str or isinstance(val[idx], tuple):
                batch[key] = val[idx]
            else:
                batch[key] = torch.tensor(val[idx]).to(device)
        return batch
