from unittest.mock import patch, mock_open

from src.data import read_data
from src.qa_training import flat_position


def test_read_data(data_jsonl):
    filepath = "path/to/jsonl/file"
    file_content = data_jsonl
    with patch("builtins.open", mock_open(read_data=file_content)):
        df = read_data(filepath)
        assert list(df.columns) == [
            "uuid",
            "title",
            "question",
            "context",
            "context_classification",
            "spoiler",
            "positions",
            "tags",
        ]


def test_flat_position():
    context = ["abc", "def", "ghi"]
    answer = "cdefg"
    pos = [[[0, 2], [2, 1]]]
    start, end = flat_position(context, pos, answer)
    assert start == 2 and end == 6
