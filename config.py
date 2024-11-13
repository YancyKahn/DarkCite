from openai_proxy import PROXY_URL, OPENAI_API_KEY, OPENAI_BASE_URL, CLAUDE_API_KEY, CLAUDE_BASE_URL

MODEL_TEMPLATE_AND_PATH = {
    "baichuan": {
        "path": "YOUR_PATH",
        "api_key": None,
        "template": "baichuan-chat",
        "type": "llm",
    },
    "llama-3-8b-instruct": {
        "path": "YOUR_PATH",
        "api_key": None,
        "template": "llama-3",
        "type": "llm",
    },
    "llama-2-7b": {
        "path": "YOUR_PATH",
        "api_key": None,
        "template": "llama-2",
        "type": "llm",
    },
    "vicuna-7b": {
        "path": "YOUR_PATH",
        "api_key": None,
        "template": "vicuna_v1.1",
        "type": "llm",
    },
    "llama-guard": {
        "path": "YOUR_PATH",
        "api_key": None,
        "template": "llama-2",
        "type": "llm"
    },
    "llama-guard-3": {
        "path": "YOUR_PATH",
        "api_key": None,
        "template": "llama-2",
        "type": "llm"
    },
    "gpt-3.5-turbo": {
        "path": PROXY_URL,
        "api_key": None,
        "template": "chatgpt",
        "type": "gpt"
    },
    "gpt-4-0613": {
        "path": PROXY_URL,
        "api_key": None,
        "template": "chatgpt",
        "type": "gpt",
    },
    "gpt-4-turbo": {
        "path": PROXY_URL,
        "api_key": None,
        "template": "chatgpt",
        "type": "gpt",
    },
    "gpt-4o": {
        "path": PROXY_URL,
        "api_key": None,
        "template": "chatgpt",
        "type": "gpt",
    },
    "gpt-4o-mini": {
        "path": PROXY_URL,
        "api_key": None,
        "template": "chatgpt",
        "type": "gpt",
    },
    "gpt-4-0125-preview": {
        "path": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "template": "chatgpt",
        "type": "gpt",
    },
    "gpt-3.5-turbo-custom": {
        "path": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "template": "chatgpt",
        "type": "gpt-custom"
    },
    "gpt-4-0613-custom": {
        "path": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "template": "chatgpt",
        "type": "gpt-custom",
    },
    "gpt-4-0125-preview-custom": {
        "path": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "template": "chatgpt",
        "type": "gpt-custom",
    },
    "gpt-4o-custom": {
        "path": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "template": "chatgpt",
        "type": "gpt-custom",
    },
    "gpt-4o-mini-custom": {
        "path": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "template": "chatgpt",
        "type": "gpt-custom",
    },
    "o1-mini-custom": {
        "path": OPENAI_BASE_URL,
        "api_key": OPENAI_API_KEY,
        "template": "chatgpt",
        "type": "gpt-custom",
    },
    "claude-3-sonnet-20240229": {
        "path": CLAUDE_BASE_URL,
        "api_key": CLAUDE_API_KEY,
        "template": "claude",
        "type": "claude"
    },
    "claude-3-5-sonnet-20240620": {
        "path": CLAUDE_BASE_URL,
        "api_key": CLAUDE_API_KEY,
        "template": "claude",
        "type": "claude"
    },
    "claude-3-haiku-20240307": {
        "path": CLAUDE_BASE_URL,
        "api_key": CLAUDE_API_KEY,
        "template": "claude",
        "type": "claude"
    },
    "claude-instant-1.2": {
        "path": CLAUDE_BASE_URL,
        "api_key": CLAUDE_API_KEY,
        "template": "claude",
        "type": "claude"
    },
    "local": {
        "path": "YOUR_PATH",
        "api_key": None,
        "template": "llama-3",
        "type": "llm",
    },
}

AGENT_CONFIG = {
    "do_sample": True,
    "temperature": 1,
    "top_p": 0.9,
    # "do_sample": False,
    "max_new_tokens": 256,
}

JUDGER_CONFIG = {
    "do_sample": True,
    "temperature": 0.9,
    "top_p": 0.9,
    "max_new_tokens": 512,
}