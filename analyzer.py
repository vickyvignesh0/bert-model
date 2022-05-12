import torch
from torch import nn
from transformers import BertModel
from datetime import datetime

MODEL_NAME = 'bert-base-cased'
MAX_LEN = 256


class SentimentClassifier(nn.Module):

    # Constructor class
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # Forward propagaion class
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #  Add a dropout layer
        output = self.drop(pooled_output)
        return self.out(output)


class Analyzer:

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['negative', 'neutral', 'positive']
        self.created_dtm = datetime.now()
        self.modified_dtm = None

    def analyze(self, reviews):
        encoded_review = self.tokenizer(
            reviews,
            max_length=MAX_LEN,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)

        output = self.model(input_ids, attention_mask)
        _, predictions = torch.max(output, dim=1)
        self.modified_dtm = datetime.now()
        return predictions

    def generate_response(self, predictions):
        total_reviews = len(predictions)
        positive_reviews = (predictions == 2).sum()
        neutral_reviews = (predictions == 1).sum()
        negative_reviews = (predictions == 0).sum()
        response = {
            "meta-data": {
                "created_dtm": str(self.created_dtm),
                "completed_dtm": str(self.modified_dtm)
            },
            "total_processed": int(total_reviews),
            "positive": int(positive_reviews),
            "negative": int(negative_reviews),
            "neutral": int(neutral_reviews)
        }
        return response
