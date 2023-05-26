import json
import re
import torch
from transformers import AutoTokenizer
from transformers import pipeline
from tqdm import tqdm


class QaModel:
    def __init__(self, model_name, num_answers=1):
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

    def predict(self, question, context):
        answer = self.model(question=question, context=context)
        return answer


def get_phrase(row, model_phrase):
    question = row.get("postText")[0]
    context = " ".join(row.get("targetParagraphs"))

    return [model_phrase.predict(question, context)["answer"]]


def get_passage(row, model_passage):
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


def get_multi(row, model_multi):
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


def predict(inputs, model_phrase, model_passage, model_multi):
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


def run_qa_model(input_file, output_file, model_name="deepset/roberta-base-squad2"):
    model = QaModel(model_name)
    model_multi = QaModel(model_name, 5)
    with open(input_file, "r") as inp, open(output_file, "w") as out:
        inp = [json.loads(i) for i in inp]

        for output in predict(inp, model, model, model_multi):
            out.write(json.dumps(output) + "\n")
