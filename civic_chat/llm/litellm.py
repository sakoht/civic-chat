
from langchain_openai import ChatOpenAI

def get_litellm_proxy(model: str) -> ChatOpenAI:
    # This lets us test accessing remote models through the LiteLLM proxy interface.
    # When using a LiteLLM we always use the OpenAI code b/c LiteLLM internally adapts to OpenAI.
    # The LiteLLM server on port 4000 exposes an OpenAI compatible interface for other models.
    return ChatOpenAI(model=model, base_url="http://0.0.0.0:4000")
