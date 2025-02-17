#!/usr/bin/env python3

"""
Set these env vars to use various services (only one required to test):
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- PERPLEXITY_API_KEY
- TOGETHER_API_KEY
"""

import asyncio
import json
import re
import time
from typing import Dict, Any, List

from langchain.agents import AgentType, Tool, initialize_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_experimental.tools.python.tool import PythonAstREPLTool


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether                                 # open-source, remote
from langchain_ollama import ChatOllama                                     #  open-source, local, replaced
from langgraph.prebuilt import create_react_agent


#
# ChatOllama does not have tool support so add this shim.
#

from langchain_experimental.llms.ollama_functions import OllamaFunctions    # open-source, local

def ChatOllama2(*args, **kwargs):
    return OllamaFunctions(*args, format="json", **kwargs)


# Set the temperature to zero for everything.
TEMP = 0

# 10s, plus one for padding for rate-limited APIs
RATE_LIMIT_DELAY = 11


def get_litellm_proxy(model: str) -> ChatOpenAI:
    # This lets us test accessing remote models through the LiteLLM proxy interface.
    # When using a LiteLLM we always use the OpenAI code b/c LiteLLM internally adapts to OpenAI.
    # The LiteLLM server on port 4000 exposes an OpenAI compatible interface for other models.
    return ChatOpenAI(model=model, base_url="http://0.0.0.0:4000")


class GraphQLAPIWrapperExtended(GraphQLAPIWrapper):
    # This override handles the problem that some models generate GQL with various wrapper text.
    def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a GraphQL query and return the results."""
        # NOTE: Some LLMs emit GQL with various quoting irregularities that mess up the default GQL tool.
        if query.startswith("```"):
            query = re.sub(r"^```.*\n", "", query)
            query = re.sub(r"```\n*$", "", query)
        elif query.startswith('{"query": '):
            query_dict = json.loads(query)
            query = query_dict['query']
        elif query.startswith('query: """'):
            query = re.sub(r'^query: """\n', "", query)
            query = re.sub(r'"""' + "\n*$", "", query)
            query = "{\n" + query + "\n}\n"
        return super()._execute_query(query)



class TogetherAPIWithDelay(ChatTogether):
    # The together.ai API rate-limits API access for some models like deepseek.
    # Use this wrapper for those models.
    def _generate(self, *args, **kwargs):
        print(f"GEN")
        result = super()._call(*args, **kwargs)  # Await the parent async method
        print("sleep...")
        time.sleep(RATE_LIMIT_DELAY)
        print("done sleeping")
        return result

    async def _agenerate(self, *args, **kwargs):
        print(f"AGEN!")
        result = await super()._agenerate(*args, **kwargs)  # Await the parent async method
        print("sleep...")
        await asyncio.sleep(RATE_LIMIT_DELAY)
        print("done sleeping")
        return result


#
# Pick an LLM for LangChain to use for the core reasoning flow.
#

## Working models

# works, fast
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=TEMP)   # civic: 29.4s, detailed
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
#llm = ChatOllama(model="deepseek-r1:8b", temperature=TEMP)
#llm = ChatOllama(model="deepseek-r1:32b", temperature=TEMP)
#llm = ChatOllama(model="ishumilin/deepseek-r1-coder-tools:8b", temperature=TEMP)
#llm = ChatOllama(model="ishumilin/deepseek-r1-coder-tools:14b", temperature=TEMP)


#
# GraphQL tools:
#

# A tool to query a Star Wars movie database for demo purposes.
starwars_graphql_wrapper = GraphQLAPIWrapperExtended(graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index")
starwars_tool = BaseGraphQLTool(
    name="Star Wars Database",
    graphql_wrapper=starwars_graphql_wrapper,
    description = """
    This tool queries a database about the Star Wars movies using GraphQL.
    """
)

# A tool to query the CIViC cancer variant database.
civic_graphql_wrapper = GraphQLAPIWrapperExtended(graphql_endpoint="https://civicdb.org/api/graphql")
civic_tool = BaseGraphQLTool(
    name="CIViC Database",
    graphql_wrapper=civic_graphql_wrapper,
    description="""
    A wrapper around the GraphQL query interface to CIViC, a catalog of genetic variants
    in genes associated with cancer.
    """
)


# A tool to run arbitrary Python code handles a ton of symbolic reasoning,
# including math and procedures that center around math, table manipulation, etc.
# In theory it has constraints to sandbox it, but this should be removed in production
# and replace w/ explicit tools with a more narrow scope.
python_repl = PythonAstREPLTool()
python_repl_tool = Tool(
    name='Python REPL',
    func=python_repl.run,
    description='''
    A Python shell. Use this to execute python commands.
    Input should be a valid python command.
    When using this tool, sometimes output is abbreviated - make sure
    it does not look abbreviated before using it in your answer.
    ''',
)

# A tool to do search.
duckduckgo = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func=duckduckgo.run,
    description='''
    A wrapper around DuckDuckGo Search.
    Useful for when you need to answer questions about current events.
    Input should be a search query.
    '''
)

#
# The following tools have two implementations, one that adds an example GQL query to the prompt,
# and the other is an encapsulated tool, which intrudes less on the prompt, and reduces errors.
#

# Disease

DISEASE_FIELDS = """
    totalCount
    pageInfo {
      hasNextPage # Is there a page after this one?
      endCursor  # If there are more pages, you can pass the param "after" with this value to get the next page.
    }
    nodes {
      id
      name
    }
"""

EXAMPLE_DISEASE_QUERY = """
# Search for the ID of a disease.
# Omit the "name" parameter to get all diseases in the system and find the best match externally.
{
  diseases(name: "Boundless Euphoria") {
    %s
  }
}
""" % DISEASE_FIELDS

@tool
def get_disease_id(disease_name: str) -> int:
    """Get the ID of a disease from the name.

    Args:
        disease_name: The name of the disease with the first letter of each word capitalized.
    """
    disease_name = disease_name.replace('"', '').lstrip().rstrip()
    disease_query = """
    {
      diseases(name: "%s") {
        %s
      }
    }
    """ % (disease_name, DISEASE_FIELDS)
    result = civic_tool._run(tool_input=disease_query)
    while isinstance(result, str):
        result = json.loads(result)
    diseases = result["diseases"]["nodes"]
    return diseases[0]["id"] if len(diseases) > 0 else None


# Molecular Profiles

MOLECULAR_PROFILE_FIELDS = """
    totalCount
    pageInfo {
      hasNextPage # Is there a page after this one?
      endCursor  # If there are more pages, you can pass the param "after" with this value to get the next page.
    }
    nodes {
      id
      name  # the full name of the profile includes gene name plus details
      description
    }
"""


EXAMPLE_MOLECULAR_PROFILE_QUERY = """
# Search for the ID of the profiles for molecular mutations by gene name.
# Omit the name to get all molecular profiles, though this is a long list and is best filtered.
{
  molecularProfiles(name: "BRCA1") {  # gene name is sufficient
    %s
  }
}
""" % MOLECULAR_PROFILE_FIELDS


@tool
def get_gene_molecular_profile_ids(gene_name: str) -> List[int]:
    """Search for the list of gene regions by gene name, containing a numeric "id", and text "name" and "description" for each.

    Args:
        gene_name: The canonical gene symbol in upper-case.
    """
    gene_name = gene_name.replace('"', '').lstrip().rstrip()
    molecular_profile_query = """
    {
      molecularProfiles(name: "%s") {  # gene name is sufficient
        %s
      }
    }
    """ % (gene_name, MOLECULAR_PROFILE_FIELDS)
    result = civic_tool._run(tool_input=molecular_profile_query)
    while isinstance(result, str):
        result = json.loads(result)
    gene_mutation_regions = result["molecularProfiles"]["nodes"]
    for region in gene_mutation_regions:
        del region["description"]
    return [v["id"] for v in gene_mutation_regions]


# Evidence

EVIDENCE_FIELDS = """
    totalCount  # Total Count of evidence entries matching the criteria
    pageInfo {
      hasNextPage # Is there a page after this one?
      endCursor  # If there are more pages, you can pass the param "after" with this value to get the next page.
    }
    nodes {
      id
      status
      molecularProfile {
        id
        name
        link
      }
      evidenceType
      evidenceLevel
      evidenceRating
      evidenceDirection
      phenotypes {
        id
        hpoId
        name
      }
      description
      disease {
        id
        doid
        name
        diseaseAliases
        displayName
      }
      therapies {
        id
        ncitId
        name
        therapyAliases
      }
      source {
        ascoAbstractId
        citationId
        pmcId
        sourceType
        title
      }
      therapyInteractionType
    }
"""

EXAMPLE_EVIDENCE_QUERY = """
# Search evidence of mutations affecting disease.
# The mutation molecularProfileId is optional, and when not supplied all are listed.
# The disease ID is optional, and if not specified all diseases will be returned.
# Supply at least one of the two parameters above to avoid returning the whole database, which is too big.
# The status of ACCEPTED ensures submissions that have not yet be found valid are ignored.
# The evidenceDirection field can be set to SUPPORTS or DOES_NOT_SUPPORT, or will return a mix of both.
{
  evidenceItems(status: ACCEPTED, diseaseId: 123, molecularProfileId:456, evidenceType: PREDICTIVE) {
    %s
  }
}
""" % EVIDENCE_FIELDS


@tool
def get_all_disease_mutations(disease_id: str) -> List[dict]:
    """Search for the list of gene mutations by disease ID, across genes.

    Args:
        disease_id: The canonical ID of the disease.
    """
    disease_id = int(disease_id.replace('"', '').rstrip())
    gql = """
        {
          evidenceItems(status: ACCEPTED, diseaseId: %d, evidenceType: PREDICTIVE) {
            %s
          }
        }
        """ % (disease_id, EVIDENCE_FIELDS)
    result = civic_tool._run(tool_input=gql)
    while isinstance(result, str):
        result = json.loads(result)
    all_predictive_mutations = result["evidenceItems"]["nodes"]
    return all_predictive_mutations


@tool
def get_disease_predictive_mutations_for_profiles(disease_id_and_gene_molecular_profile_id: str) -> List[dict]:
    """Get all predictive mutation evidence in a given disease ID and molecular profile ID from get_disease_id() and get_gene_molecular_profile_ids().

    Args:
        disease_id_and_gene_molecular_profile_id: The numeric ID of a disease in the database from get_disease_id(), then a comma, then one of the molecularProfileID from get_gene_molecular_profiles().
    """
    # NOTE: In the raw GQL version this is in the examples, but never called.  It leaves out the profile IDs and uses
    # the function above instead.
    if match := re.match("(\d+),(\d+)", disease_id_and_gene_molecular_profile_id):
        disease_id = match.group(1)
        molecular_profile_id = match.group(2)
        molecular_profile_ids = [molecular_profile_id]
    elif match := re.match("\((\d+), \[(.*)\]\)", disease_id_and_gene_molecular_profile_id):
        disease_id = match.group(1)
        molecular_profile_ids = match.group(2).split(',')
    else:
        raise Exception("Bad params!")

    disease_id = int(disease_id)

    all_predictive_mutations = []
    for molecular_profile_id in molecular_profile_ids:
        molecular_profile_id = int(molecular_profile_id)
        gql = """
            {
              evidenceItems(status: ACCEPTED, diseaseId: %d, molecularProfileId: %d, evidenceType: PREDICTIVE) {
                %s
              }
            }
            """ % (disease_id, molecular_profile_id, EVIDENCE_FIELDS)
        result = civic_tool._run(tool_input=gql)
        while isinstance(result, str):
            result = json.loads(result)
        predictive_mutations = result["evidenceItems"]["nodes"]
        all_predictive_mutations.extend(predictive_mutations)
    return all_predictive_mutations


# Combine whatever tools should be presented to the agent.
tools = [
    # comment this out to allow only the get_* tools to directly query the db.
    #civic_tool,

    # comment-out these tools to fall back to example GQL
    get_disease_id,
    get_gene_molecular_profile_ids,
    get_all_disease_mutations,
    get_disease_predictive_mutations_for_profiles,

    # use this if you want to test a smaller schema
    #starwars_tool,

    # enabling this can lead to unexpected shortcuts
    #duckduckgo_tool,

    # if this is used during testing, replace any python with custom tools with a more narrow focus
    #python_repl_tool,
]

# Include example queries wherever the related explicit tool is left out.
EXAMPLE_QUERIES = []
if get_disease_id not in tools:
    EXAMPLE_QUERIES.append(EXAMPLE_DISEASE_QUERY)
if get_gene_molecular_profile_ids not in tools:
    EXAMPLE_QUERIES.append(EXAMPLE_MOLECULAR_PROFILE_QUERY)
if get_all_disease_mutations not in tools and get_disease_predictive_mutations_for_profiles not in tools:
    EXAMPLE_QUERIES.append(EXAMPLE_EVIDENCE_QUERY)

# Construct a prompt:
prompt = "Answer the following questions by using tools if possible, followed by graphql queries, then by search.\n"
if len(EXAMPLE_QUERIES) > 0:
    prompt += (
        f"The following example queries show the primary usage pattern: {EXAMPLE_QUERIES}"
        "When no example exists, query the graphql database for the schema first, \n"
        "then write a query that complies with the schema.\n"
    )
elif civic_tool in tools or starwars_tool in tools:
    prompt += "Query the GQL schema first when no examples apply to the problem so queries match the schema.\n"

# Create an agent and app in LangChain/LangGraph.
agent_exec = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True,
)
app = create_react_agent(model=llm, tools=tools)

# For demo purposes, just ask questions directly in this script.
messages = [
    SystemMessage(prompt),
    #HumanMessage('What types of mutation molecule profiles are associated with the gene "KRAS" in relation to Colorectal Cancer?'),
    HumanMessage('What is the evidence of mutations associated with the gene "KRAS" in relation to Colorectal Cancer?'),
    #HumanMessage("What movie titles are in the Star Wars trilogy?"),
]

# Time results.
print(f"model: {llm}")
t0 = time.time()
try:
    if True:
        result = agent_exec.run(messages)
    else:
        final_state = app.invoke(
            input={"messages": messages},
            config={"configurable": {"thread_id": 42}}
        )
        result = final_state["messages"][-1].content

    t1 = time.time()
    e1 = t1 - t0
    print(result)
    print(f"success elapsed time: {e1} on model {llm}")
finally:
    t1 = time.time()
    e1 = t1 - t0
    print(f"error elapsed time: {e1} on model {llm}")


