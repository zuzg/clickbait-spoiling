import json
import re

import torch
import logging
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoBERT, Reranker
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from src.bert_classifier import predict_spoiler_class_from_text
from src.data import create_user_data, get_sentences, get_target_paragraphs


class QaModel:
    """
    Class for QA model
    """

    def __init__(self, model_name: str, num_answers: int = 1):
        """
        :param model_name: path to the model
        :param num_answers: number of expected answers
        """
        if torch.cuda.is_available():
            self.device = 0
            print("Using GPU for pipeline")
        else:
            self.device = -1
        self.model_name = model_name
        self.num_answers = num_answers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = pipeline(
            "question-answering",
            self.model_name,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=500,
            truncation=True,
            return_overflowing_tokens=True,
            stride=128,
            top_k=self.num_answers,
        )

    def predict(self, question: str, context: str) -> str:
        """
        Run prediction

        :param question: clickbait question
        :param context: context
        :return: spoiler
        """
        answer = self.model(question=question, context=context)
        return answer


def best_query(record: dict, reranker: Reranker) -> dict:
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
    return ret.text


def get_phrase(row: dict, model_phrase: QaModel) -> list:
    """
    Get results for multi phrase

    :param row: row
    :param model_phrase: model for phrase spoiler generation
    """
    question = row.get("postText")[0]
    context = " ".join(row.get("targetParagraphs"))

    return [model_phrase.predict(question, context)["answer"]]


def get_passage(row: dict, model_passage: Reranker) -> list:
    """
    Get results for passage spoiler

    :param row: row
    :param model_passage: model for passage spoiler generation
    """
    item = dict(row)
    item["question"] = row.get("postText")[0]
    item["sentences"] = get_sentences(row.get("targetParagraphs"))

    return [best_query(item, model_passage)]


def get_multi(row: dict, model_multi: QaModel) -> list:
    """
    Get results for multi spoiler

    :param row: row
    :param model_multi: model for multi spoiler generation
    """
    question = row.get("postText")[0]
    context = " ".join(row.get("targetParagraphs"))

    current_context = context
    results = []
    try:
        for _ in range(0, 5):
            candidates = model_multi.predict(question, current_context)[0]
            current_result = candidates["answer"]
            results.append(current_result)
            current_context = re.sub(current_result, "", current_context)
    except ValueError as e:
        logging.error(e)
        results = ["Error"]
    return results


def predict(
    inputs: list,
    model_phrase: QaModel,
    model_passage: Reranker,
    model_multi: QaModel,
    use_pr: bool = False,
    user: bool = False,
):
    """
    Run prediction for model

    :param inputs: list with inputs
    :param model_phrase: model for phrase generation
    :param model_passage: model for passage generation
    :param model_multi: model for multi generation
    :param use_pr: whether to use passage retrieval
    """
    for row in tqdm(inputs):
        if row.get("tags") == ["phrase"]:
            answer = get_phrase(row, model_phrase)

        elif row.get("tags") == ["passage"] and use_pr:
            answer = get_passage(row, model_passage)

        # NOTE: multi-passage won't be used while using classifier
        else:
            answer = get_multi(row, model_multi)

        yield {"uuid": row["uuid"], "spoiler": answer}


def run_inference(
    input_file: str,
    output_file: str,
    model_qa: str = "deepset/roberta-base-squad2",
    model_pr: Reranker = None,
    use_pr: bool = False,
) -> None:
    """
    Run spoiler generation

    :param input_file: input
    :param output_file: where to save generated spoilers
    :param model_qa: question answering model path
    :param model_pr: passage retrieval model path
    """
    if model_pr is None:
        model_pr = MonoBERT()

    model = QaModel(model_qa)
    model_multi = QaModel(model_qa, 5)

    with open(input_file, "r") as inp, open(output_file, "w") as out:
        inp_list = [json.loads(i) for i in inp]

        for output in predict(inp_list, model, model_pr, model_multi, use_pr):
            out.write(json.dumps(output) + "\n")


def user_inference(
    data: dict,
    model_qa: str,
    model_pr: Reranker = None,
    use_pr: bool = True,
) -> dict:
    """
    Run spoiler generation

    :data: input
    :param model_qa: question answering model path
    :param model_pr: passage retrieval model path
    """
    if model_qa is None:
        model_qa = "deepset/roberta-base-squad2"
    if model_pr is None:
        model_pr = MonoBERT()

    model = QaModel(model_qa)
    model_multi = QaModel(model_qa, 5)

    data["uuid"] = "user_input"
    output = predict([data], model, model_pr, model_multi, use_pr, user=True)
    output_list = list(output)
    return output_list[0]


def get_spoiler_from_user_input(
    postText: str,
    targetUrl: str,
    model_classification: str,
    model_qa: str = None,
    model_pr: Reranker = None,
    verbose: bool = True,
):
    """
    Run spoiler generation from the data provided by the user

    :param postText: post text
    :param targetUrl: target url
    :param model_classification: classification model path
    :param model_qa: question answering model path
    :param model_pr: passage retrieval model - reranker
    :param verbose: whether to print intermediate results
    :return: spoiler
    """
    target_paragraphs = get_target_paragraphs(targetUrl)

    if verbose:
        print("Target paragraphs (input for spoiler classification):")
        print(target_paragraphs)

    text_classification = postText + " " + (" ".join(target_paragraphs))
    prediction = predict_spoiler_class_from_text(
        text_classification, model_classification
    )
    if verbose:
        print(f"Spoiler type prediction : {prediction}")

    data = create_user_data(postText, target_paragraphs, prediction)
    if verbose:
        print("\nInput:")
        print(data)
    spoiler = user_inference(data, model_qa, model_pr)
    if verbose:
        print("Output:")
        print(spoiler)
    return spoiler
