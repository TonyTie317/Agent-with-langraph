from dotenv import load_dotenv
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langsmith"
load_dotenv()

from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

# 1. Tools
tools = [TavilySearchResults(max_results=1)]
tool_executor = ToolExecutor(tools=tools)

# 2. Prompt
prompt = hub.pull("hwchase17/structured-chat-agent")

# 3. LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
    streaming=True,
)

# 4. Agent (structured, not OpenAI function-based)
agent_runnable = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# 5. State type for LangGraph
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# 6. Run agent logic
def run_agent(data: AgentState) -> dict:
    agent_input = {
        "input": data["input"],
        "chat_history": data.get("chat_history", []),
        "intermediate_steps": data.get("intermediate_steps", []),
    }
    agent_outcome = agent_runnable.invoke(agent_input)
    return {"agent_outcome": agent_outcome}

# 7. Tool executor logic
def execute_tools(data: AgentState) -> dict:
    agent_action = data["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}

# 8. Conditional logic to continue or end
def should_continue(data: AgentState) -> str:
    return "end" if isinstance(data["agent_outcome"], AgentFinish) else "continue"

# 9. LangGraph definition
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")

app = workflow.compile()

# 10. Test input
inputs = {
    "input": "Search the latest news about Elon Musk.",
    "chat_history": []
}

# 11. Run and stream output
for step in app.stream(inputs):
    print(step)
    print("----")

    if "agent_outcome" in step and isinstance(step["agent_outcome"], AgentFinish):
        print("✅ Final output:", step["agent_outcome"].return_values.get("output"))
