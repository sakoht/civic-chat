import json
import re
from typing import Dict, Any

from langchain_community.utilities.graphql import GraphQLAPIWrapper

# This is shared by both graphql clients, and handles quirks in the different LLMs that generate
# GQL with characters that are not expected.

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
