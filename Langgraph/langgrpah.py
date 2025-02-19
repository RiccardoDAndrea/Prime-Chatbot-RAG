from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool

# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]
model = OllamaLLM(model="llama3.2:1b")  # Using the updated Ollama model

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

app = create_react_agent(model, tools, checkpointer=checkpointer)

# Use the agent
final_state = app.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config={"configurable": {"thread_id": 42}}
)

print(final_state["messages"][-1].content)
