from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun

#
# This tool does web searches, which allows it to get content that is time relevant.
# The only danger to it is that some questions that are best answered by a database query can be answered on the web.
# This means a toy example passes tests b/c of this shortcut, but a critical question requiring DB knowledge might fail.
#

duckduckgo = DuckDuckGoSearchRun()

duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func=duckduckgo.run,
    description='''
    A wrapper around DuckDuckGo Search.
    Useful for when you need to answer questions about current events.
    Input should be a search query.
    '''
)


