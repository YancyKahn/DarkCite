import sys
import os
import pandas as pd

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../..'))

import config
from utils.common import convert_json_string, save_jsonl
import argparse
from utils.language_models import LLM, load_model
import random
from fastchat.conversation import get_conv_template
import tqdm
import time
from system_template import *
import json
from judges import llama_guard_judge

def generate_citations_by_topic(args, llm, llm_judge, template_name, output_path, topic, cata="none"):
    system_prompt = get_custom_citation_system_template()
    instruction = get_citation_generate_instruction(topic)

    conv = get_conv_template(template_name)
    conv.set_system_message(system_prompt)

    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    inputs = llm.build_conversation_input_ids(prompt)

    count = 0

    reference_type_list = ["topic", "key", "citation"]

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

                    if args.judge_harmful:
                        # judge if harmful
                        types = json_context.keys()[1:-1]
                        
                        for t in types:
                            response = llama_guard_judge(args, llm_judge, None, json_context["topic"], json_context[t])
                            if response == "safe":
                                flag == False
                                break
                    
                except Exception as e:
                    print("Error: {}. {}".format(e, json_context))
                    flag = False
            
            if flag:
                count += 1
                json_context["cata"] = cata
                save_jsonl(json_context, output_path)
            else:
                print("Invalid citation: ", json_context)

def main(args):
    timestamp = str(int(time.time()))
    model_path = config.MODEL_TEMPLATE_AND_PATH[args.model_name]["path"]
    template_name = config.MODEL_TEMPLATE_AND_PATH[args.model_name]["template"]
    output_path = os.path.abspath(os.path.join(args.output_path, timestamp + '_' + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))))
    
    if args.judge_harmful:
        judge_model_name = args.judge_model
        judge_model_path = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["path"]
        judge_template_name = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["template"]
        judge_model_type = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["type"]
        judge_model_api_key = config.MODEL_TEMPLATE_AND_PATH[judge_model_name]["api_key"]
        
        llm_judge = load_model(judge_model_name, judge_model_path, judge_model_type, api_key=judge_model_api_key, device_map=args.judge_device)
    else:
        llm_judge = None
    
    model_name = args.model_name
    model_path = config.MODEL_TEMPLATE_AND_PATH[model_name]["path"]
    model_type = config.MODEL_TEMPLATE_AND_PATH[model_name]["type"]
    model_template_name = config.MODEL_TEMPLATE_AND_PATH[model_name]["template"]
    model_api_key = config.MODEL_TEMPLATE_AND_PATH[model_name]["api_key"]

    llm = load_model(model_name, model_path, model_type, api_key=model_api_key, device_map=args.device)

    if args.only_topic:
        output_path = os.path.abspath(os.path.join(args.output_path, 'tmp', timestamp + '_' + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        generate_citations_by_topic(args, llm, llm_judge, template_name, output_path, args.topic)
        
    else:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        if args.topic_path.endswith(".csv"):
            topics = pd.read_csv(args.topic_path)
            topics = topics.sample(n=len(topics))
            
            for topic in topics["goal"]:
                print("Process: ", topic)
                generate_citations_by_topic(args, llm, llm_judge, template_name, output_path, topic)
            
        elif args.topic_path.endswith(".json"):
            
            with open(args.topic_path, "r") as f:
                topics = json.load(f)

            for cls in topics:
                for topic in cls["data"]:
                    generate_citations_by_topic(args, llm, llm_judge, template_name, output_path, topic, cls["name"])
                    
        elif args.topic_path.endswith(".jsonl"):
            
            with open(args.topic_path, "r") as f:
                topics = [json.loads(line) for line in f]
                
            for item in topics:
                generate_citations_by_topic(args, llm, llm_judge, template_name, output_path, item["topic"], item["name"])
        else:
            raise Exception("Invalid topic path")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="baichuan", help='model name')
    parser.add_argument('--nums', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=2046, help='max new tokens')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--output_path', type=str, default="./outputs/citations")
    parser.add_argument('--topic_path', type=str, default="data/advbench/harmful_behaviors.csv")
    
    parser.add_argument('--topic', type=str, default="Purified water is toxic")
    parser.add_argument('--only_topic', type=bool, default=False, help='only generate one topic')
    
    parser.add_argument('--judge_model', type=str, default="llama-guard", help='judge model')
    parser.add_argument('--judge_device', type=str, default='cuda:0')
    parser.add_argument('--judge_harmful', type=bool, default=False)

    args = parser.parse_args()

    main(args)