from langchain_community.tools.graphql.tool import BaseGraphQLTool
from civic_chat.tools._gql import GraphQLAPIWrapperExtended

#
# This tool does raw queries of the civic db w/o any example queries.
# It mostly fails b/c the schema is too complicated for an LLM to create the correct queries.
#

civic_graphql_wrapper = GraphQLAPIWrapperExtended(graphql_endpoint="https://civicdb.org/api/graphql")

civic_tool = BaseGraphQLTool(
    name="CIViC Database",
    graphql_wrapper=civic_graphql_wrapper,
    description="""
    A wrapper around the GraphQL query interface to CIViC, a catalog of genetic variants
    in genes associated with cancer.
    """
)

