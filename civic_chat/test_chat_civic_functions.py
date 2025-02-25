import typer
from langchain_core.messages import HumanMessage, SystemMessage

from civic_chat.cli import create_single_inference_cli
from civic_chat.tools.civic_disease import get_disease_id
from civic_chat.tools.civic_mutation import get_gene_molecular_profile_ids
from civic_chat.tools.civic_mutation_evidence import get_all_disease_mutations, get_disease_predictive_mutations_for_profiles

tools = [
    get_disease_id,
    get_gene_molecular_profile_ids,
    get_all_disease_mutations,
    get_disease_predictive_mutations_for_profiles,
]

sys_msg = SystemMessage(
    "Answer the following questions by using tools if possible, "
    "followed by graphql queries, then by search.\n"
)

user_msg =HumanMessage('What is the evidence of mutations associated with the gene "KRAS" in relation to Colorectal Cancer?')

cli = create_single_inference_cli(tools, sys_msg, user_msg)


def test_civic_functions_as_app():
    cli(use_app=True)


def test_civic_functions_as_agent():
    cli(use_app=False)

PROMPT1 = """
Answer the following questions as best you can. You have access to the following tools:
get_disease_id(disease_name: str) -> int - Get the ID of a disease from the name.
    Args:
        disease_name: The name of the disease with the first letter of each word capitalized.
get_gene_molecular_profile_ids(gene_name: str) -> List[int] - Search for the list of gene regions by gene name, containing a numeric "id", and text "name" and "description" for each.
    Args:
        gene_name: The canonical gene symbol in upper-case.
get_all_disease_mutations(disease_id: str) -> List[dict] - Search for the list of gene mutations by disease ID, across genes.
    Args:
        disease_id: The canonical ID of the disease.
get_disease_predictive_mutations_for_profiles(disease_id_and_gene_molecular_profile_id: str) -> List[dict] - Get all predictive mutation evidence in a given disease ID and molecular profile ID from get_disease_id() and get_gene_molecular_profile_ids().
    Args:
        disease_id_and_gene_molecular_profile_id: The numeric ID of a disease in the database from get_disease_id(), then a comma, then one of the molecularProfileID from get_gene_molecular_profiles().
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [get_disease_id, get_gene_molecular_profile_ids, get_all_disease_mutations, get_disease_predictive_mutations_for_profiles]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: content='What is the evidence of mutations associated with the gene "KRAS" in relation to Colorectal Cancer?' additional_kwargs={} response_metadata={}
Thought:
"""

STOP1 = ['\nObservation:', '\n\tObservation:']

OUT1 = """
Based on the evidence, here are the key findings regarding KRAS mutations in Colorectal Cancer:

1. General KRAS Mutations:
- KRAS mutations are strongly associated with poor response to anti-EGFR therapies (cetuximab and panitumumab)
- NCCN guidelines indicate colorectal cancer patients with KRAS mutations should not be treated with anti-EGFR therapies
- Patients with KRAS mutations show significantly lower response rates (6.7% vs 35.8%) to cetuximab plus chemotherapy compared to those with wild-type KRAS

2. Specific KRAS Mutation Types:
- KRAS G12C:
  * New evidence shows promise for combination therapy with adagrasib + cetuximab (34% objective response rate)
  * Sotorasib + panitumumab shows improved progression-free survival in G12C mutated patients

- KRAS G13D:
  * Shows slightly better response to cetuximab compared to other KRAS mutations
  * However, still has generally poor outcomes with anti-EGFR therapy

3. Treatment Implications:
- KRAS mutation testing is recommended before starting anti-EGFR therapy
- Patients with KRAS mutations generally should not receive cetuximab or panitumumab
- New targeted combination approaches are showing promise for specific mutations like G12C

4. Predictive Value:
- KRAS mutation status is one of the strongest predictive biomarkers for anti-EGFR therapy response
- The presence of KRAS mutations is associated with:
  * Lower response rates
  * Shorter progression-free survival
  * Reduced overall survival when treated with anti-EGFR therapies

5. Emerging Treatments:
- New targeted combination approaches are being developed specifically for KRAS-mutated colorectal cancer
- Clinical trials are showing promising results with combination therapies targeting specific KRAS mutations

This evidence demonstrates that KRAS mutation status is a critical biomarker in colorectal cancer that significantly influences treatment decisions, particularly regarding the use of anti-EGFR therapies. The field is evolving with new targeted approaches for specific KRAS mutations showing promising results in clinical trials.
"""

if __name__ == "__main__":
    typer.run(cli)