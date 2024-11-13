import sys
import requests
import json
import os
import numpy as np
import argparse
import random
import time
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification

sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("."))
print(sys.path)

from openai_proxy import MODERATION_URL
from utils.common import convert_json_string, save_jsonl


class Toxigen():
    def __init__(self, model_name = "tomh/toxigen_roberta"):
        # roberta-base, princeton-nlp/sup-simcse-roberta-large, sentence-transformers/paraphrase-MiniLM-L12-v2
        self.model_name = model_name

        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        print("Loading model: {}".format(self.model_name))
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return self.model, self.tokenizer
    
    def get_toxicity(self, texts):
        try:
            # Use tokenizer to convert text to input tokens
            tokens = self.tokenizer(texts, return_tensors='pt', padding=True)
            # Get the last hidden state from the model
            max_tokens = 512
            tokens = {k: v[:, :max_tokens] for k, v in tokens.items()}
            
            with torch.no_grad():
                output = self.model(**tokens).logits
            # Extract the last hidden state
            prob = torch.nn.functional.softmax(output, dim=1)
            label = torch.argmax(prob, dim=1)
        except Exception as e:
            print(e)
            label = torch.zeros(len(texts))


        result = []

        for i in range(len(texts)):
            result.append({
                "text": texts[i],
                "type": str(label[i].item()),
                "toxicity": prob[i][1].item()
            })

        return result
    
def check(model, texts):
    try:
        data = texts

        texts_rebuild = []
        for text in texts:
            # if len(text) > 256:
            #     text = text[:256]
            texts_rebuild.append(text)
        result = model.get_toxicity(texts_rebuild)

    except Exception as e:
        # make sure the length of texts less than 512
        print("Error: {}".format(e))
        texts_rebuild = []
        for text in texts:
            if len(text) > 256:
                text = text[:256]
            texts_rebuild.append(text)

        result = model.get_toxicity(texts_rebuild)

    scores = []
    
    # score = generated_text["results"]
    for i, item in enumerate(result):
        scores.append({
            "text": texts[i],
            "score": item['toxicity']})
    return scores

def main(args):
    timestamp = str(int(time.time()))
    output_path = os.path.abspath(os.path.join(args.output_folder, timestamp + '_' + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)))) + ".jsonl"
    
    model = Toxigen(model_name=args.model_path)
    
    with open(args.input_file, "r") as f:    
        data = [json.loads(line) for line in f]
    
    for i in range(0, len(data), args.batch_size):
        print("Processing: {}/{}".format(i, len(data)))
        batch = data[i:i+args.batch_size]
        texts = [item["target_str"] for item in batch]
        scores = check(model, texts)
        
        for item, score in zip(batch, scores):
            item["moderation"] = score["score"]

        for item in batch:
            with open(output_path, 'a') as f:
                json.dump(item, f)
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="/data1/llm/toxigen")
    parser.add_argument("--input_file", type=str, default="/data1/workspace/yxk/FakeCiteAttack/outputs/attack/llama-2-7b/1726020229_f02wue3c/data.jsonl")
    parser.add_argument("--output_folder", type=str, default="/data1/workspace/yxk/FakeCiteAttack/outputs/moderation")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)