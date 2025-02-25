#!/usr/bin/env python3

"""
Set these env vars to use various services (only one required to test):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- PERPLEXITY_API_KEY
- TOGETHER_API_KEY
"""

import asyncio
import time

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether                                 # open-source, remote
from langchain_ollama import ChatOllama                                     # open-source, local, replaced
from langgraph.prebuilt import create_react_agent

#
# Pick an LLM for LangChain to use for the core reasoning flow.
#

from .env import TEMP

## Working models

# works, fast
#llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=TEMP)   # civic: 29.4s, detailed
# Alternative: use a LiteLLM proxy.
# llm = get_llm("anthropic/claude-3-5-sonnet-20241022")

# works, slow b/c of delay
#llm = ChatTogether(model="deepseek-ai/DeepSeek-R1") # civic 360s b/c of rate limit delay

# works, slow
#llm = Together(model="deepseek-ai/DeepSeek-V3")  # civic: 32.0s, less detailed

# fails w/ error
#llm = ChatOllama(model="chsword/DeepSeek-V3:latest")  # 3.21b, fails/error too small?

# works for graphql but reasons poorly
#llm = ChatOllama(model="phi4:14b", temperature=TEMP)


## Failing models, with reason in the coments.

# remote w/ semi-bad GQL
#llm = Together(model="Qwen/QwQ-32B-Preview")  # 23s before crash of 3rd query truncating
#llm = Together(model="Qwen/Qwen2.5-Coder-32B-Instruct")  # 3rd query tries to use an in-clause poorly

# local w/ semi-bad GQL
#llm = ChatOllama(model="mistral:7b", temperature=TEMP)

# works for star wars only
#llm = ChatOpenAI(model='gpt-4o', temperature=TEMP)  # civic: bad GQL
#llm = ChatOpenAI(model='gpt-o1', temperature=TEMP)  # civic: bad GQL

# fails
#llm = ChatOllama(model="llama3.2:3b", temperature=TEMP)   # starwars: no, civic no

# works briefly sometimes, fails often w/ wrong answer
#llm = Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")  # civic: 3.8s, very brief, sometimes wrong/empty

# crashes
#llm = ChatOllama(model="llama3.1:405b", temperature=TEMP)  # starwars: ? (not enough disk)
#llm = ChatOllama(model="llama3.3:70b", temperature=TEMP)  # starwars: ? (not enough ram)

# slow, killed before answer
#llm = ChatOllama(model="qwen2.5-coder:32b", temperature=TEMP)  # civic: very slow, killed before answer

# fails to follow langchain pattern though somq queries work
#llm = Together(model="Qwen/QwQ-32B-Preview")  # 6.8s before crash
#llm = ChatOllama(model="nezahatkorkmaz/deepseek-v3", temperature=TEMP)

# fails to write GQL
#llm = ChatOllama(model="llama3-groq-tool-use:8b", temperature=TEMP)  # civic 7.7s before exception
#llm = ChatOllama(model="deepseek-coder:6.7b", temperature=TEMP)  # civic: 18s before exception
#llm = ChatOllama(model="dolphin3:8b", temperature=TEMP)  # civic: 18s before exception
#llm = ChatOllama(model="granite3.1-dense:8b", temperature=TEMP)  # civic: 42s before error
#llm = ChatOllama(model="granite3.1-moe:3b", temperature=TEMP)  # civic: 6.4s before error
#llm = OllamaFunctions(model="deepseek-r1:14b", format="json")  # boo...

# unsupported?
#llm = ChatOllama(model="command-r7b:7b", temperature=TEMP)

# to be tested
llm = ChatOllama(model="deepseek-r1:8b", temperature=TEMP)
#llm = ChatOllama(model="deepseek-r1:32b", temperature=TEMP)
#llm = ChatOllama(model="ishumilin/deepseek-r1-coder-tools:8b", temperature=TEMP)
#llm = ChatOllama(model="ishumilin/deepseek-r1-coder-tools:14b", temperature=TEMP)

