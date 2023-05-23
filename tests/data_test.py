from unittest.mock import patch, mock_open

from src.data import read_data, save_df_to_jsonl


def test_read_data(data_jsonl):
    filepath = "path/to/jsonl/file"
    file_content = data_jsonl
    with patch("builtins.open", mock_open(read_data=file_content)):
        df = read_data(filepath)
        assert list(df.columns) == ['uuid', 'title', 'question', 'context', 'spoiler']
