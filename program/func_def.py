import torch
import numpy as np
import pandas as pd
import transformers
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AdamW, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # input_ids encodes input texts into tokenized integer sequence
        # attention_mask is used to ignore padded tokens, only attend to real tokens
        # torch.tensor(label) convert label into pytorch tensor
        encoding = self.tokenizer(
            text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        # load pre-trained bert model
        self.bert = BertModel.from_pretrained(bert_model_name)
        # initialize dropout layer to prevent overfitting, dropout rate = 0.1
        self.dropout = nn.Dropout(0.1)
        # * initialize fully connected layer, input size = bert.config.hidden_size, output size = num_classes
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # feed input_ids and attention_mask into bert model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # extract the last hidden state of the first token, which is [CLS] token
        pooled_output = outputs.pooler_output
        # feed pooled_output into dropout layer to prevent overfitting
        x = self.dropout(pooled_output)
        # feed pooled_output into fully connected layer to produce the final logits,
        # which represent the unnormalized scores for each class
        logits = self.fc(x)
        return logits


def train(model, data_loader, optimizer, scheduler, device):
    # set model to training mode
    model.train()
    for batch in data_loader:
        # PyTorch accumulates gradients, so we need to clear them out before each batch
        optimizer.zero_grad()
        # move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        # feed input_ids and attention_mask into model to get logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # calculate loss between logits and labels
        # Cross-entropy loss is commonly used for classification tasks
        # nn.CrossEntropyLoss() combines nn.LogSoftmax() and nn.NLLLoss() in a single class
        loss = nn.CrossEntropyLoss()(outputs, labels)
        # calculate gradients of model parameters with respect to loss
        loss.backward()
        # optimizer.step() performs a parameter update based on the current gradient
        optimizer.step()
        # scheduler.step() update learning rate
        scheduler.step()


def evaluate(model, data_loader, device):
    # set model to evaluation mode
    model.eval()
    predictions = []
    actual_labels = []
    # disable gradient calculation to save memory and computation
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # print(outputs)
            # torch.max() returns the maximum value of each row of the input tensor in the given dimension dim
            _, preds = torch.max(outputs, dim=1)
            # append predictions and actual labels to calculate accuracy and classification report
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions),\
        classification_report(actual_labels, predictions),\
        mean_squared_error(actual_labels, predictions)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
