import math
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


@tool
def economic_order_quantity(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Compute the economic order quantity (EOQ) in units."""
    return round(math.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit), 2)


@tool
def safety_stock(z_score: float, demand_std_per_day: float, lead_time_days: float) -> float:
    """Compute safety stock in units from a service-level z-score, daily demand std dev, and lead time."""
    return round(z_score * demand_std_per_day * math.sqrt(lead_time_days), 2)


@tool
def reorder_point(avg_daily_demand: float, lead_time_days: float, safety_stock_units: float) -> float:
    """Compute the reorder point in units."""
    return round(avg_daily_demand * lead_time_days + safety_stock_units, 2)


tools = [economic_order_quantity, safety_stock, reorder_point]
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0).bind_tools(tools)

SYSTEM = (
    "You are an inventory planning assistant. Always use the tools for calculations, "
    "never compute by hand, even simple sums; the reorder point must come from the reorder_point tool. Use z=1.65 for a 95% service level. Explain results briefly."
)


def assistant(state: MessagesState):
    messages = [{"role": "system", "content": SYSTEM}] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo"}}
    questions = [
        "A product has annual demand of 12000 units, order cost of 50, and holding cost of 2 per unit per year. What is the EOQ?",
        "Now for the same product: average daily demand is 40, lead time is 7 days, daily demand std dev is 10, and I want a 95% service level. What is the reorder point?",
        "What was the EOQ you calculated earlier?",
    ]
    for q in questions:
        result = graph.invoke({"messages": [("user", q)]}, config)
        print("Q:", q)
        print("A:", result["messages"][-1].content, "\n")
