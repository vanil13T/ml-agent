from langchain_openai import ChatOpenAI

import agent_config as config


def make_llm():
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        timeout=300,
        max_retries=2,
    )
