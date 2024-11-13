from system_template import gpt_judge_system_template, get_judge_system_prompt, get_judge_prompt
import re

def llama_guard_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs=None):
    def moderate(chat):
        input_ids = llm_judge.processor.apply_chat_template(chat, return_tensors="pt").to(llm_judge.model.device)
        output = llm_judge.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return llm_judge.processor.decode(output[0][prompt_len:], skip_special_tokens=True)

    return moderate([
        {"role": "user", "content": f"{goal}"},
        {"role": "assistant", "content": f"{target_str}"},
    ]).strip()

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
        

def local_judge(args, llm_judge, conv_judge, goal, target_str, gen_kwargs=None):
    conv = conv_judge.copy()
    # system_prompt = get_judge_system_prompt(goal)

    # conv.set_system_message(system_prompt)

    # instruction = get_judge_prompt(target_str)
    # conv.append_message(conv.roles[0], instruction)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    
    attack_prompt = gpt_judge_system_template(goal, target_str)
    
    conv.append_message(conv.roles[0], attack_prompt)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()

    inputs = llm_judge.build_conversation_input_ids(prompt)
    
    
    while True:
        response = llm_judge.generate(inputs, gen_kwargs)

        # Define a regex pattern to find the number after #thescore:
        pattern = r'#thescore:\s*(\d+)'
        
        # Search for the pattern in the text
        match = re.search(pattern, response)
        
        # If a match is found, return the number; otherwise, return None
        if match:
            return int(match.group(1))
        
    # while True:
    #     generated_text = llm_judge.generate(inputs, gen_kwargs)
    #     pattern = r'\[\[(\d+)\]\]'
    #     match = re.search(pattern, generated_text)
    #     output = int(match.group(1)) if match else None
    #     if output is None:
    #         print(f"Error in processing judge output: {generated_text}" )
    #     else:
    #         break

    return output