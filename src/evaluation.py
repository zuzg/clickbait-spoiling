from os.path import exists
from copy import deepcopy
from glob import glob
from os.path import isdir
import string
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bert_score import score
import matplotlib.pyplot as plt
from typing import Any


def error(msg: str) -> None:
    """
    Print success message nicely

    :param msg: msg to print
    """
    print("  [\033[91mx\033[0m] " + msg)
    exit(1)


def success(msg: str) -> None:
    """
    Print success message nicely

    :param msg: msg to print
    """
    print("  [\033[92mo\033[0m] " + msg)


def load_json_lines(f: str) -> list:
    """
    Read jsonl file

    :param f: filename
    :return: file as dict
    """
    if not exists(f):
        error('The file "' + f + '" does not exist.')

    ret = []
    num = 1

    if isdir(f):
        fd = glob(f + "/*.json*")
        if len(fd) != 1:
            error(
                "The input is an directory that contains multiple json files. Please create only a single json file. Got "
                + str(fd)
            )
        f = fd[0]

    with open(f, "r") as inp:
        for l in inp:
            try:
                ret += [json.loads(l)]
            except:
                error(
                    "Invalid line "
                    + str(num)
                    + ' in "'
                    + f
                    + '" with content: '
                    + l.strip()
                )
            num += 1

    success("The file " + f + " is in JSONL format.")
    return ret


def normalize_spoiler_generation(i: dict, expected_spoiler_type: str = "") -> Any:
    """
    Normalize spoiler generations 
    :param i: spoiler generations
    :param expected_spoiler_type: type of spoiler
    :return: normalized spoilers
    """
    if "uuid" not in i or "spoiler" not in i:
        error(
            "Spoiler generation does not have all required fields. Expected fields are uuid and spoiler. Got: "
            + str(i)
        )
        return

    if expected_spoiler_type and expected_spoiler_type not in i["tags"]:
        return True

    return {i["uuid"]: i["spoiler"]}


def spoiler_generations_to_map(l: list, expected_spoiler_type: Any = None) -> dict:
    """
    Transform spoiler generations 
    :param l: spoiler generations
    :param expected_spoiler_type: type of spoiler
    :return: map of spoilers
    """
    if l is None or len(l) == 0:
        error("Spoiler predictions are empty.")
    uuids = []

    for i in deepcopy(l):
        i = normalize_spoiler_generation(i, expected_spoiler_type)
        if not i:
            return {}
        elif i is True:
            continue
        uuids += list(i.keys())

    if not expected_spoiler_type and len(l) != len(set(uuids)):
        error(
            "Spoiler generations have dupliates. I found "
            + str(len(l))
            + " entries but only "
            + str(len(set(uuids)))
            + " unique uuids."
        )

    ln = [normalize_spoiler_generation(i, expected_spoiler_type) for i in l]
    ln = [i for i in ln if i and i is not True]

    success("Spoiler generations have correct format. Found " + str(len(ln)))
    ret = {}
    for i in ln:
        for k, v in i.items():
            assert k not in ret
            ret[k] = v

    return ret


def bleu_score(truth: list, prediction: list) -> float:
    """
    Calcualte BLEU score

    :param truth: true spoilers
    :param prediction: predicted spoilers
    :return: calculated score
    """
    def stopfilter(tokens):
        tmp = [token for token in tokens if token not in stopwords.words("english")]
        res = [token.lower() for token in tmp if token not in string.punctuation]
        return res

    def make_score(trut, predi):
        if len(trut) > 3 and len(predi) > 3:
            weights = (1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)
        elif len(trut) > 2 and len(predi) > 2:
            weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        elif len(trut) > 1 and len(predi) > 1:
            weights = (1.0 / 2.0, 1.0 / 2.0)
        else:
            weights = (1.0, 0.0)

        if (len(weights) == 4) and (len(trut) < 4 or len(predi) < 4):
            print(trut)
            print(predi)
            print(weights)
            print("\n")

        return sentence_bleu([trut], predi, weights=weights)
    lem_score = 0.0

    for i in range(len(truth)):
        real_answer = truth[i]
        if type(real_answer) is list:
            real_answer = " ".join(real_answer)

        pred_answer = prediction[i]
        if type(pred_answer) is list:
            pred_answer = " ".join(pred_answer)

        lem_truth_tokens = stopfilter(word_tokenize(real_answer.replace("\n", "")))
        lem_prediction_tokens = stopfilter(word_tokenize(pred_answer.replace("\n", "")))
        i_lem_score = make_score(lem_truth_tokens, lem_prediction_tokens)
        lem_score += i_lem_score

    return lem_score / len(truth)


def bert_score(truth: list, prediction: list) -> float:
    """
    Calcualte BERT score

    :param truth: true spoilers
    :param prediction: predicted spoilers
    :return: calculated score
    """
    assert len(truth) == len(prediction)
    prec, rec, f1 = score(prediction, truth, lang="en")

    return float(f1.mean())


def create_protobuf_for_task_2(actual: dict, expected: dict) -> dict:
    """
    Calculate scores for spoilers

    :param actual: predicted spoilers
    :param expected: true spoilers
    :return: dictionary with scores
    """
    keys = sorted(expected.keys())
    missing_predictions = 0

    y_true = []
    y_pred = []

    for k in keys:
        exp = expected[k]
        if type(exp) is list:
            exp = " ".join(exp)

        y_true += [exp.replace("\n", " ").strip()]

        if k in actual:
            act = actual[k]
            if type(act) is list:
                act = " ".join(act)

            y_pred += [act.replace("\n", " ").strip()]
        else:
            missing_predictions += 1
            y_pred += [""]
    return {
        "result-size": len(keys),
        "bleu-score": bleu_score(y_true, y_pred),
        "bert-score": bert_score(y_true, y_pred),
        "missing-predictions": missing_predictions,
    }


def eval_task_2(input_run: str, ground_truth_spoilers: str, output_file: str = "") -> dict:
    """
    Run evaluation on spoiler detection

    :param input_run: file with genereted spoilers
    :param ground_truth_spoilers: file with gt spoilers
    :param output_file: (optional) where to save scores
    :return: string with scores
    """
    input_run_list = load_json_lines(input_run)
    gt_spoilers_dict = load_json_lines(ground_truth_spoilers)
    input_run_dict = spoiler_generations_to_map(input_run_list)
    result_dict = dict()
    if gt_spoilers_dict == None:
        result_dict["result-size"] = len(input_run_dict.keys())
        success(
            "No ground-truth is passed. I tested the input run and the input run is valid."
        )
    else:
        for display_name, tag_name in [
            ("all-spoilers", None),
            ("phrase-spoilers", "phrase"),
            ("passage-spoilers", "passage"),
            ("multi-spoilers", "multi"),
        ]:
            print("Run evaluation for " + display_name)
            filtered_ground_truth_spoilers = spoiler_generations_to_map(
                deepcopy(gt_spoilers_dict), tag_name)

            for k, v in create_protobuf_for_task_2(
                input_run_dict, filtered_ground_truth_spoilers
            ).items():
                result_dict[f"{k}-{display_name}"] = v

    if output_file:
        with open(output_file, "w") as f:
            f.write(json.dumps(result_dict))
    return result_dict


def plot_scores(scores: dict) -> None:
    """
    Plot comparing scores and number of spoilers

    :param scores: dict with scores
    :return: none
    """
    plots = ["bleu", "bert", "size"]
    labels = ["all-spoilers", "phrase-spoilers", "passage-spoilers" , "multi-spoilers"]
    fig, axs = plt.subplots(1, 3, figsize=(12,5))
    for i, p in enumerate(plots):
        axs[i].set_title(p)
        axs[i].set_xticklabels(labels, rotation=45)
        for key, val in scores.items():
            if p in key:
                axs[i].bar(key, val)
    fig.show()
