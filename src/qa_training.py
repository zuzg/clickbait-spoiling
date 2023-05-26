from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForQuestionAnswering,
)

from .data import read_data


MODEL_CHECKPOINT = "deepset/roberta-base-squad2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def tokenize_function(dataset):
    inputs = TOKENIZER(
        dataset["question"],
        dataset["context"],
        max_length=500,
        padding="max_length",
        truncation="only_second",
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = dataset["spoiler"]
    positions = dataset["positions"]
    start_positions = []
    end_positions = []

    for i, (answer, pos, offset) in enumerate(zip(answers, positions, offset_mapping)):
        start_char = pos[0][0][0]
        end_char = start_char + len(answer[0])
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
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
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


def get_tokenized_dataset(df):
    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)


def prepare_training(train_dataset, eval_dataset):
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

    training_args = TrainingArguments(
        output_dir="./data/roberta-finetuned",
        evaluation_strategy="epoch",
        num_train_epochs=5,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=TOKENIZER,
    )
    return trainer


def finetune_roberta(train_file, eval_file):
    train_df = read_data(train_file)
    eval_df = read_data(eval_file)
    train_dataset = get_tokenized_dataset(train_df)
    validation_dataset = get_tokenized_dataset(eval_df)
    small_train_dataset = train_dataset.shuffle(seed=42).select(range(1000))
    small_eval_dataset = validation_dataset.shuffle(seed=42).select(range(200))
    trainer = prepare_training(small_train_dataset, small_eval_dataset)
    trainer.train()
    trainer.save_model("./data/roberta-finetuned")
