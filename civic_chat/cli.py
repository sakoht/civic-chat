import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentType, initialize_agent, AgentExecutor

from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph

from civic_chat.tools.duckduckgo_search import duckduckgo_tool
from civic_chat.tools.python_repl import python_repl_tool


def create_single_inference_cli(tools: list, sys_msg: SystemMessage, user_msg: HumanMessage):

    def cli(graph: bool = False, search: bool = False, code: bool = False, debug: bool = False, verbose: bool = False):
        """ The single inference CLI just processes one set of messages and prints the output.
        """
        print(f'app: {graph} search {search} code: {code} debug {debug} verbose: {verbose}')
        nonlocal tools
        if search or code:
            tools = tools.copy()
            if search:
                tools.append(duckduckgo_tool)
            if code:
                tools.append(python_repl_tool)

        from .llm_client import llm

        messages = [sys_msg, user_msg]

        print(f"Sys Message: {sys_msg}")
        print(f"User Message: {user_msg}")
        print(f"Tools: {[t.name for t in tools]}")
        print(f"LLM: {llm}")
        t0 = time.time()
        try:
            if graph:
                from langchain.globals import set_verbose, set_debug
                set_verbose(verbose)
                set_debug(debug)
                graph: CompiledGraph = create_react_agent(model=llm, tools=tools)
                stream = graph.stream(
                    input={"messages": messages},
                    config={"configurable": {"thread_id": 42}},
                    stream_mode="values"
                )
                from langchain_core.messages.base import BaseMessage
                for s in stream:
                    message: BaseMessage = s["messages"][-1]
                    if isinstance(message, tuple):
                        print(message)
                    else:
                        print(message.pretty_repr(html=True))
            else:
                from langchain.agents import create_tool_calling_agent
                #agent = create_tool_calling_agent(llm, tools, prompt_template)
                agent_exec: AgentExecutor = initialize_agent(
                    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True,
                )
                result = agent_exec.invoke(
                    {
                        'input': user_msg,
                        'chat_history': [sys_msg],
                    }
                )
            t1 = time.time()
            e1 = t1 - t0
            print(result)
            print(f"success elapsed time: {e1} on model {llm}")
        finally:
            t1 = time.time()
            e1 = t1 - t0
            print(f"error elapsed time: {e1} on model {llm}")

    return cli

