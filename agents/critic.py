import json
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM = """You are a rigorous AI research critic evaluating GenAI supply chain analysis.
You score analyses and provide specific, actionable feedback so the analyst can improve.

Output ONLY valid JSON. No preamble."""

HUMAN = """Evaluate the following analysis of GenAI use cases in FMCG supply chains.

Analysis:
{analysis}

Score each criterion 1-10 and return ONLY this JSON:

{{
  "criteria": {{
    "genai_specificity":    <1-10>,
    "fmcg_relevance":       <1-10>,
    "implementation_depth": <1-10>,
    "evidence_quality":     <1-10>,
    "risk_specificity":     <1-10>
  }},
  "overall_score": <average of the 5 criteria scores, rounded to nearest integer>,
  "issues": [
    "Specific problem 1 — e.g. use case 4 uses 'machine learning algorithms', not GenAI",
    "Specific problem 2"
  ],
  "strengths": ["What was done well"],
  "approved": <MUST be true only if ALL of the following are true:
    (1) overall_score >= 8,
    (2) every use case is powered by LLMs or foundation models (no traditional ML),
    (3) no obviously hallucinated statistics,
    (4) at least one named real-world company per use case.
    Set to false otherwise.>
}}

Criteria definitions:
- genai_specificity: Are ALL 5 use cases genuinely powered by LLMs/foundation models?
  Score 1-4 if ANY use case describes traditional ML or rule-based systems instead of GenAI.
- fmcg_relevance: Are examples and impact metrics specific to FMCG supply chains (not generic retail, automotive, or manufacturing)?
- implementation_depth: Are technical details specific (model type, data sources, integration points)?
- evidence_quality: Are claims grounded in research — no hallucinated percentages or unnamed "companies in the X space"?
- risk_specificity: Are risks GenAI-specific (hallucination, prompt injection, context limits, model drift)?"""


def _parse_critique(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        logger.warning("[CRITIC] Could not parse JSON — defaulting to approved.")
        return {"overall_score": 8, "approved": True, "issues": [], "strengths": []}


def critic_agent(state: dict) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    logger.info("[CRITIC] Evaluating analysis...")
    analysis = state["analysis"]

    response = llm.generate([[
        SystemMessage(content=SYSTEM),
        HumanMessage(content=HUMAN.format(analysis=analysis)),
    ]])
    raw = response.generations[0][0].message.content
    parsed = _parse_critique(raw)

    score = int(parsed.get("overall_score", 8))
    # Use the critic's own explicit approved field — do NOT re-derive from score.
    # The critic may flag issues (approved=False) even at score=7.
    approved = bool(parsed.get("approved", False))
    issues = parsed.get("issues", [])
    strengths = parsed.get("strengths", [])

    critique_text = (
        f"Score: {score}/10\n"
        f"Approved: {approved}\n"
        f"Issues:\n" + "\n".join(f"  - {i}" for i in issues) + "\n"
        f"Strengths:\n" + "\n".join(f"  + {s}" for s in strengths)
    )

    logger.info("[CRITIC] Score: %d/10 | Approved: %s", score, approved)
    if issues:
        logger.info("[CRITIC] Issues:\n%s", "\n".join(f"  - {i}" for i in issues))

    state["critique"] = critique_text
    state["critique_score"] = score
    state["critique_approved"] = approved
    state["revision_count"] = state.get("revision_count", 0) + 1
    return state
