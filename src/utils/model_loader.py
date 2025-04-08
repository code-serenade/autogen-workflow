# model_loader.py

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient


def get_model_client(provider: str = "ollama", model: str = "llama3.2"):
    if provider == "ollama":
        return OllamaChatCompletionClient(model=model)
    elif provider == "openai":
        return OpenAIChatCompletionClient(model=model)
    else:
        raise ValueError(f"未知模型提供者: {provider}")
