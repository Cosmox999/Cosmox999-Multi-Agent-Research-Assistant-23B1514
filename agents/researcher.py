import time
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from tools.search import search_multiple

logger = logging.getLogger(__name__)

SYSTEM = """You are a research analyst extracting structured intelligence about
Generative AI deployments in FMCG supply chains from raw web search results.

Your output will be used by a downstream analyst agent to identify the top 5 GenAI use cases.
Write only factual summaries drawn from the source material. Do not invent examples."""

HUMAN = """Below are raw search results from multiple queries about GenAI in FMCG supply chains.

{raw_sources}

Extract and organise the key findings under these headings:
1. Confirmed real-world deployments (company name, use case, outcomes if stated)
2. Technology patterns (which GenAI tech: LLMs, RAG, generative models, multimodal?)
3. Supply chain areas covered (demand, procurement, logistics, inventory, risk, NPD…)
4. Measurable outcomes or KPIs mentioned
5. Vendors / platforms mentioned (e.g. Microsoft, Google, SAP, startups)

Be specific. Reference source titles where possible. Do not speculate beyond what the sources say."""


def researcher_agent(state: dict) -> dict:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    queries = state.get("search_queries", [])
    if not queries:
        logger.warning("[RESEARCHER] No search queries found — using raw query.")
        queries = [state["query"]]

    logger.info("[RESEARCHER] Running %d searches...", len(queries))
    sources, raw_combined = search_multiple(queries)

    logger.info("[RESEARCHER] Summarising findings with LLM...")
    response = llm.generate([[
        SystemMessage(content=SYSTEM),
        HumanMessage(content=HUMAN.format(raw_sources=raw_combined[:12000])),  # token guard
    ]])
    research_summary = response.generations[0][0].message.content

    state["sources"] = sources
    state["research"] = research_summary
    logger.info("[RESEARCHER] Done. Research summary length: %d chars", len(research_summary))
    logger.info("[RESEARCHER] Pausing 3s to avoid downstream rate limits...")
    time.sleep(3)
    return state
