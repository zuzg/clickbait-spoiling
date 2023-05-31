import json
import re
from collections.abc import Generator

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


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


def get_phrase(row: dict, model_phrase: QaModel) -> list:
    """
    Get results for multi phrase

    :param row: row
    :param model_phrase: model for phrase spoiler generation
    """
    question = row.get("postText")[0]
    context = " ".join(row.get("targetParagraphs"))

    return [model_phrase.predict(question, context)["answer"]]


def get_passage(row: dict, model_passage: QaModel) -> list:
    """
    Get results for passage spoiler

    :param row: row
    :param model_passage: model for passage spoiler generation
    """
    question = row.get("postText")[0]
    context = " ".join(row.get("targetParagraphs"))

    answer = model_passage.predict(question, context)["answer"]

    candidates = []
    for sentence in context.split("."):
        if answer in sentence:
            candidates.append(sentence.strip())

    if not candidates:
        # print("No candidates found")
        return [""]
    elif len(candidates) == 1:
        return [candidates[0]]
    elif len(candidates) > 1:
        # print("Multiple candidates found")
        return [candidates[0]]
    return [""]


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
    except:
        # print("Error generating multipart spoiler")
        results = ["Error"]
    return results


def predict(
    inputs: list, model_phrase: QaModel, model_passage: QaModel, model_multi: QaModel
) -> Generator:
    """
    Run prediction for model

    :param inputs: list with inputs
    :param model_phrase: model for phrase generation
    :param model_passage: model for passage generation
    :param model_multi: model for multi generation
    """
    for row in tqdm(inputs):
        if row.get("tags") == ["phrase"]:
            answer = get_phrase(row, model_phrase)

        elif row.get("tags") == ["passage"]:
            answer = get_passage(row, model_passage)

        elif row.get("tags") == ["multi"]:
            answer = get_multi(row, model_multi)
        else:
            # print("Tag not found")
            raise NotImplemented

        yield {"uuid": row["uuid"], "spoiler": answer}


def run_inference(
    input_file: str,
    output_file: str,
    model_qa: str = "deepset/roberta-base-squad2",
    model_pr: str = "",
) -> None:
    """
    Run spoiler generation

    :param input_file: input
    :param output_file: where to save generated spoilers
    :param model_qa: question answering model path
    :param model_pr: passage retrieval model path
    """
    model = QaModel(model_qa)
    # TODO add passage retrieval model here
    model_multi = QaModel(model_qa, 5)
    with open(input_file, "r") as inp, open(output_file, "w") as out:
        inp_list = [json.loads(i) for i in inp]

        for output in predict(inp_list, model, model, model_multi):
            out.write(json.dumps(output) + "\n")
