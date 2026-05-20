import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM = """You are a research planning agent specialising in Generative AI and FMCG supply chains.
Your job is to produce a list of precise web search queries that will surface real,
evidence-based examples of LLMs and foundation models being deployed in FMCG supply chains.

Rules:
- Queries must be specific enough to return industry case studies, vendor whitepapers, or analyst reports.
- Each query should target a DIFFERENT area of the supply chain (demand, procurement,
  logistics, inventory, supplier risk, etc.).
- Prefer queries that will find 2024-2025 deployments.
- Do NOT produce generic queries like "AI in supply chain".
- Output ONLY a JSON array of exactly 5 strings. No preamble, no explanation."""

HUMAN = """Topic: {query}

Return a JSON array of 5 targeted search queries.

Example format (do not copy these — generate relevant ones):
["LLM demand sensing Unilever FMCG 2024",
 "generative AI procurement negotiation consumer goods 2024",
 "GPT supply chain disruption early warning FMCG",
 "LLM natural language inventory replenishment CPG 2024",
 "generative AI supplier risk assessment P&G Nestlé 2024"]"""


def _parse_queries(text: str) -> list:
    """Extract a JSON list from the model response, with fallback."""
    text = text.strip()
    # Try direct parse
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    # Fallback: sensible hard-coded queries that produce real results
    logger.warning("Planner JSON parse failed — using fallback queries.")
    return [
        "LLM generative AI demand forecasting FMCG consumer goods 2024",
        "generative AI supply chain visibility procurement FMCG 2024 2025",
        "LLM natural language inventory optimisation CPG retail 2024",
        "generative AI supplier risk management Unilever Nestle P&G 2024",
        "foundation model new product development FMCG market intelligence 2024",
    ]


def planner_agent(state: dict) -> dict:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    logger.info("[PLANNER] Generating targeted search queries...")
    query = state["query"]

    response = llm.generate([[
        SystemMessage(content=SYSTEM),
        HumanMessage(content=HUMAN.format(query=query)),
    ]])
    raw = response.generations[0][0].message.content
    queries = _parse_queries(raw)

    logger.info("[PLANNER] Search queries: %s", queries)
    state["search_queries"] = queries
    state["revision_count"] = 0
    return state
