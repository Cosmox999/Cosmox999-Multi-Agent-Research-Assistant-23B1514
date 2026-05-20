import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM = """You are a senior FMCG supply chain strategist and Generative AI expert.
You produce structured, evidence-based analysis for executive audiences.

DEFINITION — what counts as Generative AI:
  QUALIFIES: LLMs (GPT-4, Llama, Gemini), RAG pipelines, fine-tuned foundation models,
             multimodal vision-language models, generative diffusion models,
             LLM-based agents, natural language interfaces over ERP/WMS data.
  DISQUALIFIED: Random forests, XGBoost, ARIMA, traditional neural nets (LSTM, CNN),
                rule-based engines, classical optimisation (OR-Tools, linear programming),
                generic "AI" or "machine learning" without a foundation model.

STRICT RULES — failure on ANY rule causes the critic to reject the analysis:
1. Every use case MUST use only QUALIFYING GenAI technology above.
   Before writing each use case, mentally ask: "Is there an LLM or foundation model here?"
   If the answer is no, replace it with a different use case.
2. Each use case must cover a DIFFERENT supply chain area — no two use cases on demand.
3. Implementation must name the model type (e.g. RAG over SAP data, fine-tuned Llama-3,
   GPT-4 with function calling) and a real data source (ERP, POS, supplier EDI).
4. Maturity must be exactly one of: Pilot | Emerging | Scaling | Mainstream.
5. Risks must be GenAI-specific: hallucination, prompt injection, context window limits,
   model drift, training data leakage — NOT generic risks like "data quality" alone.
6. real_examples must name actual companies or vendors. Write "Evidence pending" if none found.
   Never write "companies in the X space" — that is a hallucination.
7. Output ONLY valid JSON. No preamble, no markdown fences, no extra text."""

HUMAN_FIRST = """Research findings:
{research}

Before writing JSON, silently verify each use case against these checks:
  [ ] Does it use an LLM or foundation model (not traditional ML)?
  [ ] Is the supply chain area different from the other 4 use cases?
  [ ] Does real_examples name an actual company or say "Evidence pending"?

Then produce ONLY the following JSON. No extra text before or after.

{{
  "use_cases": [
    {{
      "rank": 1,
      "name": "Short descriptive name (e.g. LLM-Powered Demand Signal Interpretation)",
      "genai_technology": "Exact GenAI tech — e.g. RAG pipeline over ERP + POS data using Llama-3",
      "supply_chain_area": "Specific area — e.g. Demand Sensing, Procurement, Logistics, NPD",
      "implementation": "3-4 sentences: which foundation model, what data it reads, how it integrates with existing systems (ERP/WMS/TMS), and what the output looks like",
      "expected_impact": "Specific outcome — cite a source figure if available, otherwise describe the mechanism",
      "risks": "2-3 GenAI-specific risks with one sentence each",
      "maturity": "Pilot | Emerging | Scaling | Mainstream",
      "real_examples": ["Nestle: deployed X for Y outcome", "Evidence pending"]
    }}
  ]
}}"""

HUMAN_REVISION = """Research findings:
{research}

Your PREVIOUS analysis was scored {score}/10 and NOT approved.
The critic identified these specific problems:
{critique}

Instructions for revision:
- For each issue mentioning a specific use case (e.g. "use case 4 uses traditional ML"):
  REPLACE that use case entirely with a new one from a different supply chain area.
- Do not just reword the problem use case — replace it with a genuinely different GenAI application.
- Ensure EVERY use case passes: uses an LLM or foundation model, names real companies or "Evidence pending",
  and covers a supply chain area not already covered by another use case.

Output ONLY valid JSON using the exact same schema as before. No preamble."""


def _parse_analysis(text: str) -> str:
    """Return cleaned JSON string; fallback to raw text if parsing fails."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json.loads(text[start:end])   # validate
        return text[start:end]
    except (ValueError, json.JSONDecodeError):
        logger.warning("[ANALYST] Output is not valid JSON — storing raw text.")
        return text


def analyst_agent(state: dict) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    research = state["research"]
    critique = state.get("critique", "")
    score = state.get("critique_score", 10)
    revision = state.get("revision_count", 0)

    if critique and revision > 0:
        logger.info("[ANALYST] Revision %d — incorporating critic feedback (score was %d/10)...", revision, score)
        prompt = HUMAN_REVISION.format(research=research, score=score, critique=critique)
    else:
        logger.info("[ANALYST] First-pass analysis...")
        prompt = HUMAN_FIRST.format(research=research)

    response = llm.generate([[
        SystemMessage(content=SYSTEM),
        HumanMessage(content=prompt),
    ]])
    raw = response.generations[0][0].message.content
    state["analysis"] = _parse_analysis(raw)
    logger.info("[ANALYST] Analysis complete (%d chars).", len(state["analysis"]))
    return state
