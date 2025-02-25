import json
from typing import List

from langchain_core.tools import tool

from .civic_db_gql import civic_tool


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
