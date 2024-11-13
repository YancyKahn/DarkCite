import sys
import os
import pandas as pd

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../..'))

import config
from utils.common import convert_json_string, save_jsonl
import argparse
from utils.language_models import load_model
import random
from fastchat.conversation import get_conv_template
import tqdm
import time
from system_template import *
import json

def classify(args, llm, template_name, output_path, topic):
    system_prompt = get_harmful_type_classification_system_prompt(topic)
    inputs = llm.build_conversation_input_ids(system_prompt)

    count = 0

    reference_type_list = ["cata", "name"]

    while count < args.nums:

        print("Generating scene {}/{}: {}".format(count+1, args.nums, topic))

        generated_text = llm.generate(inputs, config.AGENT_CONFIG)

        json_context = convert_json_string(generated_text)

        flag = True
        if json_context is not None:

            for rf_type in reference_type_list:
                try:
                    if rf_type not in json_context.keys():
                        flag = False
                        break
                    if json_context[rf_type] == "":
                        flag = False
                        break
                except Exception as e:
                    print("Error: {}. {}".format(e, json_context))
                    flag = False
            
            if flag:
                count += 1
                json_context["topic"] = topic
                json_context["cata"] = int(json_context["cata"])
                save_jsonl(json_context, output_path)
            else:
                print("Invalid citation: ", json_context)

def main(args):
    timestamp = str(int(time.time()))
    model_path = config.MODEL_TEMPLATE_AND_PATH[args.model_name]["path"]
    template_name = config.MODEL_TEMPLATE_AND_PATH[args.model_name]["template"]
    model_type = config.MODEL_TEMPLATE_AND_PATH[args.model_name]["type"]
    output_path = os.path.abspath(os.path.join(args.output_path, timestamp + '_' + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))))
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    llm = load_model(args.model_name, model_path, model_type, device_map=args.device)

    if args.topic_path.endswith(".csv"):
        topics = pd.read_csv(args.topic_path)
        topics = topics.sample(n=len(topics))
        
        for topic in topics["goal"]:
            print("Process: ", topic)
            classify(args, llm, template_name, output_path, topic)
    elif args.topic_path.endswith(".jsonl"):
        with open(args.topic_path, "r") as f:
            topics = [json.loads(line) for line in f]
            
        for topic in topics:
            goal = topic["topic"]
            print("Process: ", goal)
            classify(args, llm, template_name, output_path, goal)
    else:
        raise Exception("Invalid topic path")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='model name')
    parser.add_argument('--nums', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new tokens')
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--output_path', type=str, default="./outputs/classification")
    parser.add_argument('--topic_path', type=str, default="data/advbench/harmful_behaviors.csv")

    args = parser.parse_args()

    main(args)