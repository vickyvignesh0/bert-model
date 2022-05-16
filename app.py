import torch
import json
from flask import Flask, request
from transformers import BertTokenizer

from analyzer import Analyzer, SentimentClassifier

MODEL_NAME = 'bert-base-cased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['negative', 'neutral', 'positive']
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = SentimentClassifier(len(class_names))
model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
model = model.to(device)

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hi there! everything working fine..'


@app.route('/ping', methods=['POST', 'GET'])
def ping():
    req = request.json
    return f"This is a post-request ping check dear {req['name']} and everything works fine here as well"


@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    req = json.loads(request.json)
    review, resp = None, {}

    if req:
        reviews = req['reviews']
    if reviews:
        bert_analyzer = Analyzer(
            tokenizer=tokenizer,
            model=model
        )
        predictions = bert_analyzer.analyze(
            reviews,
        )
        resp = bert_analyzer.generate_response(predictions)
    return resp


app.run(host='0.0.0.0', port=8080)
