from langchain_community.tools.graphql.tool import BaseGraphQLTool

from .civic_db_gql import civic_graphql_wrapper
from .civic_disease import DISEASE_FIELDS
from .civic_mutation import MOLECULAR_PROFILE_FIELDS
from .civic_mutation_evidence import EVIDENCE_FIELDS


#
# This tool does queries of the civic db, but with examples on hand.
# Unlike the raw connection, it gets correct results.
# The downside is that the queries are so big the examples take up a ton of space in the context.
#

EXAMPLE_DISEASE_QUERY = """
# Search for the ID of a disease.
# Omit the "name" parameter to get all diseases in the system and find the best match externally.
{
  diseases(name: "Boundless Euphoria") {
    %s
  }
}
""" % DISEASE_FIELDS


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

EXAMPLE_MOLECULAR_PROFILE_QUERY = """
# Search for the ID of the profiles for molecular mutations by gene name.
# Omit the name to get all molecular profiles, though this is a long list and is best filtered.
{
  molecularProfiles(name: "BRCA1") {  # gene name is sufficient
    %s
  }
}
""" % MOLECULAR_PROFILE_FIELDS


EXAMPLE_QUERIES = [
    EXAMPLE_DISEASE_QUERY,
    EXAMPLE_MOLECULAR_PROFILE_QUERY,
    EXAMPLE_EVIDENCE_QUERY
]

civic_tool_with_example_queries = BaseGraphQLTool(
    name="CIViC Database",
    graphql_wrapper=civic_graphql_wrapper,
    description=f"""
    A wrapper around the GraphQL query interface to CIViC, a catalog of genetic variants
    in genes associated with cancer.  Use the following example queries as templates:
    {EXAMPLE_QUERIES}
    """
)
