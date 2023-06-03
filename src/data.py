import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from typing import List
from src.inference import Row


def get_sentences(paragraphs: List[str]) -> List[str]:
    """
    Get sentences from paragraphs.
    """
    sentences = list()
    for paragraph in paragraphs:
        for sentence in sent_tokenize(paragraph):
            sentences += [sentence]
    return sentences


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
                "question": " ".join(i["postText"]),  # query
                "context": i["targetParagraphs"],  # paragrahps
                "sentences": get_sentences(i["targetParagraphs"]),
                "context_classification": " ".join(i["postText"])
                + " "
                + (" ".join(i["targetParagraphs"])),
                "spoiler": " ".join(i["spoiler"]),
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
                "context": " ".join(i["postText"])
                + " "
                + (" ".join(i["targetParagraphs"])),
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
    spoilers = df[["uuid", "spoiler"]]
    json_output = spoilers.to_json(orient="records", lines=True)
    with open(filepath, "w") as f:
        f.write(json_output)


def clean_text(text: str) -> str:
    """
    Clean text from email addresses, urls and extra spaces.

    :param text: text to clean
    :return: cleaned text
    """
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def get_target_paragraphs(link: str) -> List[str]:
    """
    Get target paragraphs from the provided link.

    :param link: link to the article
    :return: list of target paragraphs
    """
    max_retries = 5

    for retry in range(max_retries):
        response = requests.get(link)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")

            target_paragraphs = []
            for p in paragraphs:
                cleaned_text = clean_text(p.get_text())
                if len(cleaned_text) > 0:
                    target_paragraphs.append(cleaned_text)
            return target_paragraphs
        elif response.status_code == 429:
            # rate limit exceeded, wait for a few seconds...
            wait_time = 5
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            return []
    return []


def create_user_data(postText: str, target_paragraphs: List[str], prediction: str) -> Row:
    """
    Create user data from provided postText, target_paragraphs and prediction.

    :param postText: postText from the user
    :param target_paragraphs: target_paragraphs from the user
    :param prediction: prediction from the model
    :return: dictionary with postText, target_paragraphs and prediction
    """
    data = Row()
    data["postText"] = postText
    data["targetParagraphs"] = target_paragraphs
    data["tags"] = [prediction]
    return data
