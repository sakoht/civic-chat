import json

from langchain_core.tools import tool

from .civic_db_gql import civic_tool

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
