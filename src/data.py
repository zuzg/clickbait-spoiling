import json
import pandas as pd


def read_data(filename: str) -> pd.DataFrame:
    """
    Read data into dataframe from provided filename

    :param filename: name of the file to load
    :return: dataframe with contents of the provided file
    """
    data_json = [json.loads(i) for i in open(filename, "r")]
    df = pd.DataFrame(
        [
            {
                "uuid": i["uuid"],
                "title": i["targetTitle"],
                "question": " ".join(i["postText"]),
                "context": i["targetParagraphs"],
                "context_classification": " ".join(i["postText"]) + " " + (" ".join(i["targetParagraphs"])),
                "spoiler": i["spoiler"],
                "positions": i["spoilerPositions"],
                "tags": 1 if i["tags"][0].lower() == "phrase" else 0,
            }
            for i in data_json
        ]
    )
    return df


def read_spoilers(filename: str) -> pd.DataFrame:
    """
    Read data into dataframe with uuid and spoiler from provided filename

    :param filename: name of the file to load
    :return: dataframe with uuid and spoiler
    """
    data_json = [json.loads(i) for i in open(filename, "r")]
    df = pd.DataFrame(
        [
            {
                "uuid": i["uuid"],
                "spoiler": i["spoiler"],
            }
            for i in data_json
        ]
    )
    return df


def read_data_classification(filename: str) -> pd.DataFrame:
    """
    Read data into dataframe from provided filename

    :param filename: name of the file to load
    :return: dataframe with contents of the provided file
    """
    data_json = [json.loads(i) for i in open(filename, "r")]
    df = pd.DataFrame(
        [
            {
                "context": " ".join(i["postText"]) + " " + (" ".join(i["targetParagraphs"])),
                "tags": 1 if i["tags"][0].lower() == "phrase" else 0,
            }
            for i in data_json
        ]
    )
    return df


def save_df_to_jsonl(df: pd.DataFrame, filepath: str) -> None:
    """
    Save dataframe as jsonl file

    :param df: dataframe with uuid and spoiler columns
    :param filepath: where to save jsonl file
    :return: None
    """
    spoilers = df[["uuid","spoiler"]]
    json_output = spoilers.to_json(orient='records', lines=True)
    with open(filepath, 'w') as f:
        f.write(json_output)
