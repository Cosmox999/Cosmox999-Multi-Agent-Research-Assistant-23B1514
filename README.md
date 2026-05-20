# Multi-Agent GenAI Research Assistant

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2+-1C1C1C?style=flat"/>
  <img src="https://img.shields.io/badge/Groq-Llama--3.1--8B-F55036?style=flat"/>
  <img src="https://img.shields.io/badge/Tavily-Search_API-4A90E2?style=flat"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat"/>
</p>

> An autonomous multi-agent AI pipeline that researches any topic end-to-end — planning queries, retrieving live web sources, analysing findings, self-critiquing quality, and generating a structured final report — all in under 25 seconds.

---

## Problem Statement

Research and knowledge synthesis is time-consuming and fragmented. Analysts must manually search multiple sources, evaluate quality, and compile structured reports. This project replaces that workflow with a fully autonomous agent pipeline — search, analyse, critique, revise, report — without human intervention.

---

## Why This Project Matters

- Demonstrates **agentic AI architecture** with a real cyclic graph (not just a linear chain)
- Implements **LLM-as-judge** critique loop with automated revision — a production pattern used in enterprise AI systems
- Achieves **~25 second** end-to-end runtime on free-tier APIs (Groq + Tavily)
- Includes **graceful degradation** for search failures — production-ready fault tolerance

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐   generates 5 queries   ┌──────────────────────┐
│   PLANNER   │ ──────────────────────► │   Search Queries     │
└─────────────┘                         └──────────────────────┘
                                                   │
                                                   ▼
┌─────────────┐     Tavily Search API    ┌──────────────────────┐
│ RESEARCHER  │ ◄──────────────────────► │   19 Web Sources     │
└─────────────┘                          └──────────────────────┘
       │
       ▼
┌─────────────┐
│   ANALYST   │ ◄──────────────────────────────────────┐
└─────────────┘                                        │ revision loop
       │                                               │ (if score < 8)
       ▼                                               │
┌─────────────┐   score < 8   ┌────────────────────────┴───┐
│    CRITIC   │ ─────────────►│  Critique + Score (0–10)   │
└─────────────┘               └────────────────────────────┘
       │ score ≥ 8
       ▼
┌─────────────┐
│  REPORTER   │ ──► results/sample_output.md
└─────────────┘
```

All inter-agent communication flows through a single **TypedDict state object** managed by LangGraph's `StateGraph`. The critic-revision loop is a **conditional edge** — LangGraph routes back to the Analyst automatically if quality is below threshold.

---

## Agent Roles

| Agent | Responsibility | Key Output |
|-------|---------------|------------|
| **Planner** | Decomposes the research query into 5 targeted search queries | `search_queries: list[str]` |
| **Researcher** | Runs Tavily searches, deduplicates results, summarises with LLM | `sources: list`, `research: str` |
| **Analyst** | Produces structured analysis; revises based on critic feedback | `analysis: str` |
| **Critic** | Scores output quality (0–10), flags issues, routes for revision | `critique_score: int`, `critique_approved: bool` |
| **Reporter** | Formats and writes the final structured report | `final_report: str` |

---

## Features

- **5-agent autonomous pipeline** orchestrated with LangGraph StateGraph
- **Critic-driven revision loop** — automatic quality control with configurable threshold (score ≥ 8/10)
- **Live web research** — Tavily retrieves and deduplicates 15–20 sources per run
- **~25 second runtime** using Groq-hosted Llama-3.1-8B (free tier)
- **Graceful degradation** — pipeline continues if individual searches fail
- **Edge case demo script** — simulates search timeout to prove fault tolerance

---

## Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Agent Orchestration | LangGraph | Native cyclic graph + conditional edges for revision loop |
| LLM | Groq + Llama-3.1-8B-Instant | ~10× faster than OpenAI; free tier; sufficient context |
| Web Search | Tavily API | Purpose-built for LLM agents; structured JSON output |
| State Management | TypedDict (in-memory) | Zero-dependency; fully inspectable at every step |
| Language | Python 3.10+ | Broadest LLM/agent ecosystem |

---

## Results

```
Sources retrieved  : 19
Critic score       : 8 / 10
Revision cycles    : 2
Pipeline runtime   : 22.5 seconds
Final report size  : ~10,500 characters
```

See [`results/sample_output.md`](./results/sample_output.md) for the full generated report.

---

## Dataset / Input

No dataset required. The pipeline takes a **natural language research query** as input and fetches live web sources via the Tavily Search API at runtime.

**Demo query used:**
> *"Identify the top 5 emerging GenAI use cases in FMCG supply chains. For each use case, describe implementation approach, expected impact, risks, and maturity level."*

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Cosmox999/Cosmox999-Multi-Agent-Research-Assistant-23B1514.git
cd Cosmox999-Multi-Agent-Research-Assistant-23B1514

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Open .env and fill in your TAVILY_API_KEY and GROQ_API_KEY
```

**Free API keys:**
- Tavily: [app.tavily.com](https://app.tavily.com)
- Groq: [console.groq.com](https://console.groq.com)

---

## Usage

**Run the full pipeline:**
```bash
python main.py
```

The pipeline plans queries → retrieves sources → analyses → critiques → revises if needed → writes the final report to `results/sample_output.md`.

**Change the research topic** — edit `QUERY` in `main.py`:
```python
QUERY = """
Your research question here.
"""
```

**Run the edge case demo:**
```bash
python demo_edge_case.py
```
Simulates a `ConnectionError` on query 3 of 5 — pipeline continues gracefully with remaining sources.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Tavily search timeout / error | Warning logged; pipeline continues with remaining sources |
| LLM returns malformed JSON | Raw text stored as fallback; corrected in next revision |
| Groq rate limit | 3-second pause inserted after search phase |
| Critic never approves | Revision capped at `MAX_REVISIONS`; Reporter runs on best output |

---

## Project Structure

```
├── main.py                    # Entry point — builds and streams the agent graph
├── demo_edge_case.py          # Simulates search failure; proves graceful degradation
├── requirements.txt
├── .env.example               # API key template
├── LICENSE
│
├── agents/
│   ├── state.py               # Shared TypedDict state schema
│   ├── planner.py             # Query decomposition agent
│   ├── researcher.py          # Web search + summarisation agent
│   ├── analyst.py             # Structured analysis + revision agent
│   ├── critic.py              # LLM-as-judge scoring agent
│   └── reporter.py            # Final report generation agent
│
├── tools/
│   └── search.py              # Tavily search wrapper with deduplication
│
├── workflow/
│   └── graph.py               # LangGraph StateGraph and conditional routing logic
│
└── results/
    └── sample_output.md       # Example report generated by the pipeline
```

---

## Future Improvements

- **Persistent memory** — add ChromaDB / FAISS for incremental cross-session research
- **Citation grounding** — enforce citation-required output schema to reduce hallucination risk
- **Parallelised search** — run all 5 Tavily queries concurrently with `asyncio`
- **Streaming UI** — wrap in Streamlit or Gradio for interactive use
- **Structured revision prompts** — inject critic issues as JSON constraints for targeted revisions

---

## Author

**Ganesh Pandurang Sonawane**
Indian Institute of Technology Bombay

[![GitHub](https://img.shields.io/badge/GitHub-Cosmox999-181717?style=flat&logo=github)](https://github.com/Cosmox999)
