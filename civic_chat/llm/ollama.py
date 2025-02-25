

#
# ChatOllama did not have tool support originally w/o this shium.
#

from langchain_experimental.llms.ollama_functions import OllamaFunctions    # open-source, local

def ChatOllamaWithFunctionShim(*args, **kwargs):
    return OllamaFunctions(*args, format="json", **kwargs)
