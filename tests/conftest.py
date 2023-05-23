import pytest
import pandas as pd


@pytest.fixture
def data_jsonl():
    data = {
        "uuid": ["uuid"],
        "title": ["title"],
        "postText": ["postText"],
        "targetTitle": ["targetTitle"],
        "targetParagraphs": ["targetParagraphs"],
        "spoiler": ["spoiler"],
        "blabla": ["blablabla"],
    }
    df = pd.DataFrame.from_dict(data)
    jsonl = df.to_json(orient="records", lines=True)
    return jsonl
