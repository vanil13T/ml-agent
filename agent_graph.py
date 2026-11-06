from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

import agent_config as config
import agent_llm as llm_module
import agent_nodes as nodes
from agent_state import AgentState
from agent_tools import tools


def route_after_generate(state: dict) -> str:
    last = state["messages"][-1]
    if state.get("llm_failures", 0) >= 3:
        return END
    if not isinstance(last, AIMessage):
        return "generate"
    if last.tool_calls:
        return "tools"
    if "ФИНАЛ" in last.content.upper():
        return END
    if state.get("nudges", 0) >= config.MAX_NUDGES or config.time_left() < 120:
        return END
    return "nudge"


def router(state: dict) -> str:
    if config.time_left() < 120:
        return END
    if state.get("experiment_num", 0) >= config.MAX_EXP:
        return END
    if state.get("fails_in_row", 0) >= config.MAX_FAILS_IN_ROW:
        return END
    return "generate"


def build_graph():
    llm_with_tools = llm_module.make_llm().bind_tools(tools, parallel_tool_calls=False)

    def generate(state: dict) -> dict:
        return nodes.generate_code_node(state, llm_with_tools)

    builder = StateGraph(AgentState)
    builder.add_node("analyze", nodes.analyze_task_node)
    builder.add_node("generate", generate)
    builder.add_node("nudge", nodes.nudge_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("evaluate", nodes.evaluate_node)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "generate")
    builder.add_conditional_edges("generate", route_after_generate, ["tools", "nudge", "generate", END])
    builder.add_edge("nudge", "generate")
    builder.add_edge("tools", "evaluate")
    builder.add_conditional_edges("evaluate", router, ["generate", END])
    return builder.compile()
