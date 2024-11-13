import sys
import os

from utils.common import convert_json_string, save_jsonl
from system_template import *
from utils.language_models import LLM, load_model

from judges import llama_guard_judge, gpt_judge, local_judge
from fastchat.conversation import get_conv_template
import random
import pandas as pd
import tqdm
import argparse
import pandas as pd
import config
import json
import time
import re
import torch

def load_model_and_conversation(args, target_model_name, judge_model_name):
    target_model_path = config.MODEL_TEMPLATE_AND_PATH[target_model_name]["path"]
    target_template_name = config.MODEL_TEMPLATE_AND_PATH[target_model_name]["template"]
    target_model_type = config.MODEL_TEMPLATE_AND_PATH[target_model_name]["type"]
    target_model_apikey = config.MODEL_TEMPLATE_AND_PATH[target_model_name]["api_key"]

    judge_model_path = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["path"]
    judge_template_name = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["template"]
    judge_model_type = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["type"]
    judge_model_apikey = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["api_key"]

    if target_model_name == judge_model_name:
        # llm = LLM(target_model_name, target_model_path, args.target_device)
        llm = load_model(target_model_name, target_model_path, target_model_type, api_key=target_model_apikey, device_map=args.target_device)

        return llm, llm, get_conv_template(target_template_name), get_conv_template(target_template_name)
    else:
        llm_target = load_model(target_model_name, target_model_path, target_model_type, api_key=target_model_apikey, device_map=args.target_device)
        llm_judge = load_model(judge_model_name, judge_model_path, judge_model_type, api_key=judge_model_apikey, device_map=args.judge_device)

        return llm_target, llm_judge, get_conv_template(target_template_name), get_conv_template(judge_template_name)


def generate_target(args, llm_target, conv_target, key, goal, citation, citation_type, gen_kwargs=None):
    if "auth" == args.defense:
        system_prompt = defense_auth_system_template()
    elif "harm" == args.defense:
        system_prompt = defense_harm_system_template()
    elif "auth-harm" == args.defense:
        system_prompt = defense_auth_and_harm_system_template()
    else:
        system_prompt = None
        
    if "gpt" in args.target_model or "claude" in args.target_model:
        prompt = get_attack_prompt(key, goal, citation, style=citation_type)
    else:
        conv = conv_target.copy()

        user_instruction = get_attack_prompt(key, goal, citation, style=citation_type)
        
        if args.defense != "none":
            # conv.set_system_message(system_prompt)
            conv.system_message = system_prompt
            
        conv.append_message(conv.roles[0], user_instruction)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
    

    inputs = llm_target.build_conversation_input_ids(prompt, system_prompt)

    result = []
    count = 0
    while count < args.nums_pre_scene:

        generated_text = llm_target.generate(inputs, gen_kwargs)

        result.append({"target_str": generated_text,
                       "prompt": prompt})
        count += 1

    return result

def judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs=None):
    if  "llama-guard" in args.judge_model:
        output = llama_guard_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs)
    elif "gpt" in args.judge_model:
        output = gpt_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs)
    elif args.judge_model == "local":
        output = local_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs)

    return output

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer.encode(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss.item()

    perplexity = torch.exp(torch.tensor(loss))
    return perplexity.item()

def main(args):
    timestamp = str(int(time.time()))
    output_path = os.path.abspath(os.path.join(args.output_path, args.target_model, timestamp + '_' + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))))
    
    print(output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    llm_target, llm_judge, conv_target, conv_judge = load_model_and_conversation(args, args.target_model, args.judge_model)

    with open(args.input_path, "r") as f:
        input_list = [json.loads(line) for line in f.readlines()]
        
    if args.sub_nums == -1:
        args.sub_nums = len(input_list)
    
    if args.sub_nums < len(input_list):
        # 随机取sub_nums个
        input_list = random.sample(input_list, args.sub_nums)


    gen_kwargs_target = config.AGENT_CONFIG
    gen_kwargs_judger = config.JUDGER_CONFIG

    default_citation_list = list(input_list[0].keys())
    
    if "citation" not in default_citation_list:
        # 去掉topic和cata
        default_citation_list = list(set(default_citation_list) - {"topic", "cata", "key"})
    else:
        default_citation_list = ["citation"]
        
    print(default_citation_list)
    
        
    if args.topic == "all":
        citation_list = default_citation_list
    else:
        citation_list = [args.topic]
        
    if args.multiple_choice:
        citation_list = ["none"]
    else:
        citation_list = default_citation_list

    for item in tqdm.tqdm(input_list[args.start_index:]):
        for citation_type in citation_list:
            goal = item["topic"]
            key =  item["key"] if "key" in item.keys() else goal 
            # key = goal
            
            if citation_type == "citation":
                citation = item["citation"]["content"]
                citation_type = item["citation"]["type"]
            else:
                if args.multiple_choice:
                    citation = ""
                    for i in range(len(default_citation_list)):
                        citation += f"{i + 1}. {item[default_citation_list[i]]}\n"
                else:
                    citation = f"1. {item[citation_type]}"
            
            cnt = 3
            
            while cnt > 0:                
                try:
                    target_response_list = generate_target(args, llm_target, conv_target, key, goal, citation, citation_type, gen_kwargs_target)
                    break
                except Exception as e:
                    print(e)
                    cnt -= 1
            
            for response in target_response_list:
                target_str = response["target_str"]
                prompt = response["prompt"]
                judgement = judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs_judger)
                target_model_type = config.MODEL_TEMPLATE_AND_PATH[args.target_model]["type"]

                if target_model_type == "llm":
                    ppl = calculate_perplexity(llm_target.model, llm_target.processor, prompt)

                result = {
                    "model": args.target_model,
                    "judge": args.judge_model,
                    "cata": item["cata"] if "cata" in item else "none",
                    "citation_type": citation_type,
                    "goal": goal,
                    "key": key,
                    "citation": citation,
                    "prompt": prompt,
                    "target_str": target_str,
                    "ppl": ppl if target_model_type == "llm" else None,
                    "judgement": judgement,
                }

                save_jsonl(result, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_model", type=str, default="llama-2-7b")
    parser.add_argument("--target_device", type=str, default="cuda:0")
    parser.add_argument('--nums_pre_scene', type=int, default=3)
    parser.add_argument("--sub_nums", type=int, default=-1)
    
    parser.add_argument('--topic', type=str, default="all")

    parser.add_argument('--judge_model', type=str, default='llama-guard', choices=["llama-guard", "gpt-4-turbo", "o1-mini-custom", "gpt-4-0613", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo-custom", "gpt-4o-mini-custom", "llama-guard-3", "claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620", "local"])
    parser.add_argument('--judge_device', type=str, default='cuda:1')

    parser.add_argument('--multiple_choice', action='store_true', default=False)

    parser.add_argument('--output_path', type=str, default="./outputs/attack")
    parser.add_argument('--input_path', type=str, default="data/fakeCite/data-new.jsonl")
    
    parser.add_argument("--mode", type=str, default="custom", choices=["auto", "one-by-one", "custom"])

    parser.add_argument("--start_index", type=int, default=0)
    
    parser.add_argument("--defense", type=str, default="none", choices=["none", "auth", "harm", "auth-harm"])

    args = parser.parse_args()
    main(args)