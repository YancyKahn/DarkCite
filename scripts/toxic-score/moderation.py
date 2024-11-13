import sys
import requests
import json
import os
import numpy as np
import argparse
import random
import time

sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("."))
print(sys.path)

from openai_proxy import MODERATION_URL
from utils.common import convert_json_string, save_jsonl

def check(texts):
    payload = json.dumps({
        "input": texts,
    })

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", MODERATION_URL, headers=headers, data=payload)

    generated_text = response.json()
    
    scores = []
    
    # score = generated_text["results"]
    for i, item in enumerate(generated_text["results"]):
        scores.append({
            "text": texts[i],
            "score": np.max(list(item["category_scores"].values())).max()})
    return scores

def main(args):
    timestamp = str(int(time.time()))
    output_path = os.path.abspath(os.path.join(args.output_folder, timestamp + '_' + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)))) + ".jsonl"
    
    with open(args.input_file, "r") as f:    
        data = [json.loads(line) for line in f]
    
    for i in range(0, len(data), args.batch_size):
        print("Processing: {}/{}".format(i, len(data)))
        batch = data[i:i+args.batch_size]
        texts = [item["target_str"] for item in batch]
        scores = check(texts)
        
        for item, score in zip(batch, scores):
            item["moderation"] = score["score"]

        for item in batch:
            with open(output_path, 'a') as f:
                json.dump(item, f)
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", type=str, default="/data1/workspace/yxk/FakeCiteAttack/outputs/attack/llama-2-7b/1726020229_f02wue3c/data.jsonl")
    parser.add_argument("--output_folder", type=str, default="/data1/workspace/yxk/FakeCiteAttack/outputs/moderation")
    parser.add_argument("--batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    main(args)