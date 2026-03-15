from typing import TypedDict, List


class AgentState(TypedDict):
    # Input
    query: str

    # Planner output: targeted Tavily search queries
    search_queries: List[str]

    # Researcher output: raw source objects + summarised text
    sources: List[dict]   # [{title, url, content}]
    research: str

    # Analyst output: structured JSON string (parsed downstream)
    analysis: str

    # Critic output
    critique: str
    critique_score: int    # 1-10 numeric score
    critique_approved: bool  # explicit approval gate from critic JSON
    revision_count: int    # safety cap: max 2 revision loops

    # Final output
    final_report: str
