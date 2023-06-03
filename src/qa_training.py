import pandas as pd
from datasets import Dataset
from transformers.models.auto.modeling_auto import AutoModelForQuestionAnswering
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from .data import read_data

MODEL_CHECKPOINT = "deepset/roberta-base-squad2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def flat_position(context: list, pos: list, answer: str) -> tuple:
    """
    Map paragraph positions to flat positions

    :param context: context of the answer
    :param pos: paragraph style position
    :param answer: spoiler
    :return: flat start and flat end
    """
    start_p = pos[0][0][0]
    flat_start = pos[0][0][1]
    count = 0
    for i in range(start_p):
        count += len(context[i])
    flat_start += count
    flat_end = flat_start + len(answer) - 1
    return flat_start, flat_end


def tokenize_function(dataset: Dataset) -> dict:
    """
    Tokenize dataset

    :param dataset: dataset to be tokenized
    :return: tokenized dataset
    """
    inputs = TOKENIZER(
        dataset["question"],
        dataset["flat_context"],
        max_length=500,
        padding="max_length",
        truncation="only_second",
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = dataset["spoiler"]
    positions = dataset["positions"]
    contexts = dataset["context"]
    start_positions = []
    end_positions = []

    for i, (context, answer, pos, offset) in enumerate(
            zip(contexts, answers, positions, offset_mapping)):
        start_char, end_char = flat_position(context, pos, answer[0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if (offset[context_start][0] > end_char or
                offset[context_end][1] < start_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def get_tokenized_dataset(df: pd.DataFrame) -> Dataset:
    """
    Tokenize dataset

    :param df: dataframe to tokenize
    :return: tokenized dataset
    """
    df["flat_context"] = ["".join(p) for p in df["context"]]
    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)


def prepare_training(train_dataset: Dataset, eval_dataset: Dataset) -> Trainer:
    """
    Prepare training

    :param train_dataset: train dataset
    :param train_eval: eval dataset
    :return: trainer object
    """
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

    training_args = TrainingArguments(
        output_dir="./data/roberta-finetuned",
        evaluation_strategy=IntervalStrategy.EPOCH,
        num_train_epochs=20,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=TOKENIZER,
    )
    return trainer


def get_only_phrases(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to consist only phrases type of spoiler

    :param dataset: dataset to filter
    :return: filtered dataset
    """
    return dataset.loc[dataset.tags == 1]


def finetune_roberta(train_file: str, eval_file: str) -> None:
    """
    Perform RoBERTa finetuning on custom data

    :param train_file: path to train dataset
    :param eval_file: path to eval dataset
    """
    train_df = get_only_phrases(read_data(train_file))
    eval_df = get_only_phrases(read_data(eval_file))
    train_dataset = get_tokenized_dataset(train_df)
    validation_dataset = get_tokenized_dataset(eval_df)
    # small_train_dataset = train_dataset.select(range(1000))
    # small_eval_dataset = validation_dataset.select(range(200))
    trainer = prepare_training(train_dataset.shuffle(
        seed=42), validation_dataset.shuffle(seed=42))
    trainer.train()
    trainer.save_model("./data/roberta-finetuned")
