import json
import re
from typing import List

from langchain_core.tools import tool

from .civic_db_gql import civic_tool


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
