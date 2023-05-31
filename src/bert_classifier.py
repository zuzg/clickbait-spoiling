import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm


class BertDataset(Dataset):
    """
    Dataset class for Clickbait Spoiler Classification
    """

    def __init__(self, tokenizer, train_df, max_length=512):
        """
        :param tokenizer: BertTokenizer
        :param train_df: Pandas DataFrame
        :param max_length: Maximum length of the input sequence
        """
        super(BertDataset, self).__init__()
        self.train_df = train_df
        self.tokenizer = tokenizer
        self.target = train_df["tags"]
        self.max_length = max_length

    def __len__(self):
        return len(self.train_df)

    def process_input(self, text, index):
        """
        :param text: Input text
        :param index: Index of the text in the dataset
        :return: Dictionary of input_ids, attention_mask, token_type_ids and target
        """
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(self.target[index], dtype=torch.long),
        }

    def __getitem__(self, index):
        text = self.train_df.iloc[index]["text"]
        return self.process_input(text, index)


class BERTClassifier(nn.Module):
    """
    BERT Classifier for Clickbait Spoiler Classification
    """

    def __init__(self, model_checkpoint):
        super(BERTClassifier, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained(model_checkpoint)
        self.dropout1 = nn.Dropout(0.1)  # regularization
        self.fc1 = nn.Linear(768, 512)  # fully connected layer
        self.dropout2 = nn.Dropout(0.1)  # regularization
        self.fc2 = nn.Linear(512, 256)  # fully connected layer
        self.relu = nn.ReLU()  # non-linearity
        self.fc3 = nn.Linear(256, 1)  # output

    def forward(self, ids, mask, token_type_ids):
        _, x = self.bert_model(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)
        return out


def finetune_BERT(
    epochs, dataloader, model, loss_fn=nn.BCEWithLogitsLoss(), optimizer=None
):
    """
    Finetune BERT model for Clickbait Spoiler Classification
    :param epochs: Number of epochs to train
    :param dataloader: DataLoader
    :param model: BERTClassifier
    :param loss_fn: Loss function
    :param optimizer: Optimizer
    :return: Finetuned BERT model
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # freeze BERT parameters
    for param in model.bert_model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)

    for epoch in range(epochs):
        correct = 0
        total = 0

        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch, dl in loop:
            ids = dl["ids"].to(device)
            token_type_ids = dl["token_type_ids"].to(device)
            mask = dl["mask"].to(device)
            label = dl["target"].to(device)
            label = label.unsqueeze(1)

            optimizer.zero_grad()

            output = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()

            pred = np.where(output.cpu() >= 0, 1, 0)

            correct += sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            total += pred.shape[0]
            accuracy = correct / total

            loop.set_description(
                f"Epoch={epoch}/{epochs} (accuracy: {correct}/{total})"
            )
            loop.set_postfix(loss=loss.item(), acc=accuracy)

    return model


def prepare_input_bert_classifier(input_text, tokenizer, max_length=512):
    """
    Prepare input for BERT Classifier

    :param input_text: Input text
    :param tokenizer: BertTokenizer
    :param max_length: Maximum length of the input sequence
    :return: Dictionary of input_ids, attention_mask and token_type_ids
    """
    inputs = tokenizer.encode_plus(
        input_text,
        None,
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=max_length,
        truncation=True,
    )

    ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    mask = inputs["attention_mask"]

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
    }


def predict_spoiler_class_from_text(
    text: str, model: BERTClassifier, model_checkpoint: str = "bert-base-uncased"
):
    """
    Predict spoiler class from text

    :param text: Input text
    :param model: Finetuned BERTClassifier
    :param model_checkpoint: Model checkpoint
    :return: 1 if 'phrase', 0 if 'passage'/'multi'
    """
    tokenizer = transformers.BertTokenizer.from_pretrained(model_checkpoint)
    input = prepare_input_bert_classifier(text, tokenizer)
    output = model(
        input["ids"].unsqueeze(0),
        input["mask"].unsqueeze(0),
        input["token_type_ids"].unsqueeze(0),
    ).item()
    return "phrase" if output >= 0 else "passage"
