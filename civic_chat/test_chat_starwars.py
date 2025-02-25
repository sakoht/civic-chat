import typer
from langchain_core.messages import HumanMessage, SystemMessage

from civic_chat.cli import create_single_inference_cli
from civic_chat.tools.starwars_gql import starwars_tool

tools = [starwars_tool]

messages = [
    SystemMessage(
        "Answer the following questions by using tools if possible, followed by graphql queries, then by search.\n"
        "Query the GQL schema first when no examples apply to the problem so queries match the schema.\n"
    ),
    HumanMessage("What movie titles are in the Star Wars trilogy?"),
]

cli = create_single_inference_cli(tools, messages)


def test_civic_gql_as_app():
    cli(use_app=True)


def test_civic_gql_as_agent():
    cli(use_app=False)


if __name__ == "__main__":
    typer.run(cli)
