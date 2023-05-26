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
        "blabla": ["blablabla"],
        "context_classification": ["asdf"],
        "spoiler": ["spoiler"],
        "spoilerPositions": ["spoilerPositions"],
        "tags": ["phrase"],
    }
    df = pd.DataFrame.from_dict(data)
    jsonl = df.to_json(orient="records", lines=True)
    return jsonl
