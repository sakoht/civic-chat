from langchain_community.tools.graphql.tool import BaseGraphQLTool
from civic_chat.tools._gql import GraphQLAPIWrapperExtended

# A tool to query a Star Wars movie database for demo purposes.
# This is just to demo that an LLM _can_ generate correct queries on a small and simple schema.

starwars_graphql_wrapper = GraphQLAPIWrapperExtended(graphql_endpoint="https://swapi-graphql.netlify.app/.netlify/functions/index")

starwars_tool = BaseGraphQLTool(
    name="Star Wars Database",
    graphql_wrapper=starwars_graphql_wrapper,
    description = """
    This tool queries a database about the Star Wars movies using GraphQL.
    """
)
