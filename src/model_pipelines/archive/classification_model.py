import dill
import numpy as np
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class SpoilerClassificationModel:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.metric = load_metric("accuracy")

    def tokenize_function(self, dataset):
        return self.tokenizer(dataset["context"], padding="max_length", truncation=True)

    def tokenize_datasets(self, training_dataset, eval_dataset):
        train_dataset = Dataset.from_pandas(training_dataset)
        eval_dataset = Dataset.from_pandas(eval_dataset)

        tokenization_function = dill.dumps(self.tokenize_function)

        self.train_dataset = train_dataset.map(
            dill.loads(tokenization_function), batched=True
        )
        self.eval_dataset = eval_dataset.map(
            dill.loads(tokenization_function), batched=True
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def prepare_training(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=2
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            # logging_dir="./logs",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        return trainer, model
