import logging
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.critic import critic_agent
from agents.reporter import report_agent

logger = logging.getLogger(__name__)


def _should_revise(state: AgentState) -> str:
    """
    Conditional edge after the critic.

    Route to 'analyst' (revision loop) if:
      - critique_approved is False (critic explicitly rejected it), AND
      - revision_count < 2 (safety cap to prevent infinite loops)

    Otherwise route to 'reporter'.
    The approved flag is set by the critic and reflects qualitative judgement,
    not just a score threshold — this prevents a score of exactly 7 with known
    issues from slipping through to the reporter.
    """
    approved = state.get("critique_approved", True)
    revisions = state.get("revision_count", 0)
    score = state.get("critique_score", 10)

    if not approved and revisions < 2:
        logger.info(
            "[GRAPH] Critic NOT approved (score %d/10) — routing back to analyst (revision %d).",
            score, revisions,
        )
        return "analyst"

    logger.info(
        "[GRAPH] Critic approved (score %d/10). Routing to reporter (revisions: %d).",
        score, revisions,
    )
    return "reporter"


def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("planner",    planner_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst",    analyst_agent)
    workflow.add_node("critic",     critic_agent)
    workflow.add_node("reporter",   report_agent)

    # Linear flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner",    "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst",    "critic")

    # Conditional reflection loop: critic → analyst (revise) OR reporter (done)
    workflow.add_conditional_edges(
        "critic",
        _should_revise,
        {
            "analyst":  "analyst",
            "reporter": "reporter",
        },
    )

    workflow.add_edge("reporter", END)

    return workflow.compile()
