import typer
from langchain_core.messages import HumanMessage, SystemMessage

from civic_chat.cli import create_single_inference_cli
from civic_chat.tools.civic_db_gql_with_examples import civic_tool_with_example_queries

tools = [civic_tool_with_example_queries]

messages = [
    SystemMessage(
        "Answer the following questions by using tools if possible, followed by graphql queries, then by search.\n"
        "When no example exists, query the graphql database for the schema first, \n"
        "then write a query that complies with the schema.\n"
        "Query the GQL schema first when no examples apply to the problem so queries match the schema.\n"
    ),
    HumanMessage('What is the evidence of mutations associated with the gene "KRAS" in relation to Colorectal Cancer?'),
]

cli = create_single_inference_cli(tools, messages)


def test_civic_gql_as_app():
    cli(use_app=True)


def test_civic_gql_as_agent():
    cli(use_app=False)


if __name__ == "__main__":
    typer.run(cli)
