import sys
import requests
import json
import os
import numpy as np
import argparse
import random
import time
import re


sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("."))

from utils.common import convert_json_string, save_jsonl
from system_template import *
from utils.language_models import LLM, load_model

from judges import llama_guard_judge, gpt_judge, local_judge
from fastchat.conversation import get_conv_template

import config
from openai_proxy import MODERATION_URL
from utils.common import convert_json_string, save_jsonl

def load_model_and_conversation(args, judge_model_name):

    judge_model_path = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["path"]
    judge_template_name = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["template"]
    judge_model_type = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["type"]
    judge_model_apikey = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["api_key"]

    llm_judge = load_model(judge_model_name, judge_model_path, judge_model_type, api_key=judge_model_apikey, device_map=args.judge_device)

    return llm_judge, get_conv_template(judge_template_name)


def gpt_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs=None):
    judge_prompt = gpt_judge_system_template(goal, target_str)

    prompt = llm_judge.build_conversation_input_ids(judge_prompt)

    while True:
        response = llm_judge.generate(prompt, gen_kwargs)

        # Define a regex pattern to find the number after #thescore:
        pattern = r'#thescore:\s*(\d+)'
        
        # Search for the pattern in the text
        match = re.search(pattern, response)
        
        # If a match is found, return the number; otherwise, return None
        if match:
            return int(match.group(1))
        
def main(args):
    timestamp = str(int(time.time()))
    output_path = os.path.abspath(os.path.join(args.output_folder, timestamp + '_' + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))))
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    gen_kwargs_judger = config.JUDGER_CONFIG
    llm_judge, conv_judge = load_model_and_conversation(args, args.judge_model)
    
    with open(args.input_file, "r") as f:    
        data = [json.loads(line) for line in f]
    
    for i in range(len(data)):
        print("Processing: {}/{}".format(i, len(data)))
        item = data[i]
        goal = item["goal"]
        target_str = item["target_str"]
        
        output = judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs_judger)
        item["new_judgement"] = output
        item["new_judger"] = args.judge_model
        
        save_jsonl(item, output_path)


def judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs=None):
    if  "llama-guard" in args.judge_model:
        output = llama_guard_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs)
    elif "gpt" in args.judge_model:
        output = gpt_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs)
    elif args.judge_model == "local":
        output = local_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs)

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", type=str, default="/data1/workspace/yxk/FakeCiteAttack/outputs/attack/llama-2-7b/1726020229_f02wue3c/data.jsonl")
    parser.add_argument("--output_folder", type=str, default="/data1/workspace/yxk/FakeCiteAttack/outputs/moderation")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--judge_model', type=str, default='llama-guard', choices=["llama-guard", "gpt-4-0613", "gpt-4o", "gpt-4o-mini", "gpt-4o-mini-custom", "llama-guard-3", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620", "local"])
    parser.add_argument('--judge_device', type=str, default='cuda:1')
    
    args = parser.parse_args()
    
    main(args)