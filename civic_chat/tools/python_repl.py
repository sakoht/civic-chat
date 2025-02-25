from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import Tool

# A tool to run arbitrary Python code handles a ton of symbolic reasoning,
# including math and procedures that center around math, table manipulation, etc.

# In theory it has constraints to sandbox it, but this should be removed in production
# and replace w/ explicit tools with a more narrow scope.

python_repl = PythonAstREPLTool()

python_repl_tool = Tool(
    name='Python REPL',
    func=python_repl.run,
    description='''
    A Python shell. Use this to execute python commands.
    Input should be a valid python command.
    When using this tool, sometimes output is abbreviated - make sure
    it does not look abbreviated before using it in your answer.
    ''',
)
