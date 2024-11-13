import argparse
import ast
import os
import json
import random
def calculate_ratio(expression):
    
    try:
        result = eval(expression)
        return result
    
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid expression: {e}")
    

def convert_json_string(json_string):
    try:
        json_string = json_string.strip()
        json_object = ast.literal_eval(json_string)
        if json_object is None:
            return None
        if isinstance(json_object, str):
            return json_object
        return json_object
    except (ValueError, SyntaxError) as e:
        print(f"Error(Not JSON Format): {e}, {json_string}")
        return None
    
def save_jsonl(jsondata, path):
    save_path = os.path.join(path, "data.jsonl")

    with open(save_path, 'a') as f:
        json.dump(jsondata, f)
        f.write('\n')