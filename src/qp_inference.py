import pandas as pd

from pygaggle.rerank.base import Query, Reranker, Text


def best_query(record: pd.Series, reranker: Reranker) -> dict:
    """
    Find the best query for a record.

    :param record: record
    :param reranker: reranker (MonoBERT or MonoT5)
    :return: dict with uuid and spoiler
    """
    passages = record["sentences"]
    passages = zip(range(len(passages)), passages)
    documents = [Text(i[1], {"docid": i[0]}, 0) for i in passages]
    ret = sorted(
        reranker.rerank(Query(record["question"]), documents),
        key=lambda i: i.score,
        reverse=True,
    )[0]
    return {"uuid": record["uuid"], "spoiler": ret.text}
