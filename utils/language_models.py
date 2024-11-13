from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import requests

class GPT():

    def __init__(self, model_name, base_url, api_key=None):
        self.modle_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    def build_conversation_input_ids(self, text, system_template=None):
        if system_template is not None:
            messages = [{
                "role": "system",
                "content": system_template
            },
            {
                "role": "user",
                "content": text
            }            
            ]
        else:
            messages = [{
                        "role": "user",
                        "content": text
                    }]
        
        return messages

    def generate(self, prompt, gen_kwargs):
        payload = json.dumps({
            "model": self.modle_name,   #gpt-4o,gpt-4-turbo,gpt-4
            "messages": prompt,
            "temperature": gen_kwargs["temperature"],
        })

        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key != None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.request("POST", self.base_url, headers=headers, data=payload)
        
        try:
            generated_text = response.json()["choices"][0]["message"]["content"]
        except:
            print(response.text)

        return generated_text

class Claude():
    def __init__(self, model_name, base_url, api_key=None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    def build_conversation_input_ids(self, text, system_template=None):
        if system_template is not None:
            messages = [{
                "role": "system",
                "content": system_template
            },
            {
                "role": "user",
                "content": text
            }]
        else:
            messages = [{
                "role": "user",
                "content": text
            }]
        return messages

    def generate(self, prompt, gen_kwargs):
        payload = json.dumps({
            "model": self.model_name,  # 例如 "claude-1" 或其他
            "messages": prompt,
            "temperature": gen_kwargs.get("temperature", 1.0),
            "max_tokens": gen_kwargs.get("max_new_tokens", 256),
        })

        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(self.base_url, headers=headers, data=payload)

        if response.status_code == 200:
            generated_text = response.json()["content"][0]["text"]
            return generated_text
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

class LLM():
    def __init__(self, model_name, model_path, device_map="auto"):
        self.model_name = model_name
        self.model_path = model_path
        self.device_map = device_map

        self.model, self.processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, 
                                                  use_fast=False, 
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path,                 
                                                     torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True,
                                                    device_map=self.device_map,
                                                    trust_remote_code=True)

        return model, tokenizer
    
    def build_conversation_input_ids(self, text, system_template=None):
        inputs = self.processor(text, return_tensors="pt").to(self.model.device)

        return inputs
    
    def generate(self, inputs, gen_kwargs=None):
        if gen_kwargs == None:
            gen_kwargs = {
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.9,
                }
        
        outputs = self.model.generate(**inputs, **gen_kwargs)

        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_texts = self.processor.decode(outputs[0], skip_special_tokens=True)

        return generated_texts


def load_model(model_name, model_path, model_type, api_key=None, device_map="auto"):
    if model_type == "gpt":
        return GPT(model_name, model_path, api_key=api_key)
    elif model_type == "gpt-custom":
        model_name = "-".join(model_name.split("-")[:-1])
        return GPT(model_name, model_path, api_key=api_key)
    elif model_type == "claude":
        return Claude(model_name, model_path, api_key=api_key)
    elif model_type == "llm":
        return LLM(model_name, model_path, device_map=device_map)
    else:
        raise ValueError("Invalid model type")