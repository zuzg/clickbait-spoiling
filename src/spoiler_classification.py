import numpy as np
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_CHECKPOINT = "microsoft/deberta-base"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def tokenize_function(dataset):
    return TOKENIZER(dataset["context"], padding="max_length", truncation=True)


def get_tokenized_dataset(df):
    dataset = Dataset.from_pandas(df)
    return dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def prepare_training(train_dataset, eval_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
    )
    # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer, model
