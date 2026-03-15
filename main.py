import logging
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup — INFO to console, DEBUG available via LOG_LEVEL=DEBUG env var
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from workflow.graph import build_graph   # noqa: E402 (import after logging setup)


QUERY = """
Identify the top 5 emerging GenAI use cases in FMCG supply chains.
For each use case describe implementation approach, expected impact,
risks and maturity level.
"""

DIVIDER = "=" * 70


def main():
    logger.info(DIVIDER)
    logger.info("  GenAI FMCG Supply Chain Research Agent — starting")
    logger.info(DIVIDER)

    app = build_graph()

    initial_state = {
        "query":            QUERY,
        "search_queries":   [],
        "sources":          [],
        "research":         "",
        "analysis":         "",
        "critique":         "",
        "critique_score":   0,
        "critique_approved": False,
        "revision_count":   0,
        "final_report":     "",
    }

    t0 = time.time()

    # stream_mode="values" yields the FULL state after every node completes.
    # The last yielded value is the final state — no second invoke() needed.
    logger.info("Invoking agent graph...")
    logger.info(DIVIDER)

    result = initial_state
    for result in app.stream(initial_state, stream_mode="values"):
        # Identify which node just finished by checking which keys changed
        completed = [
            k for k in ("search_queries", "research", "analysis", "critique", "final_report")
            if result.get(k) and result[k] != initial_state.get(k)
        ]
        node_hint = completed[-1].upper() if completed else "..."
        logger.info("[STEP COMPLETE] state updated → %-20s", node_hint)

    elapsed = time.time() - t0
    logger.info(DIVIDER)
    logger.info("Pipeline finished in %.1f seconds.", elapsed)
    logger.info(DIVIDER)

    report = result.get("final_report", "No report generated.")

    # Print to console
    print("\n" + DIVIDER)
    print("FINAL REPORT")
    print(DIVIDER)
    print(report)

    # Also save to file so it's easy to share / inspect
    output_path = "final_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", output_path)

    # Print run metadata
    sources = result.get("sources", [])
    score = result.get("critique_score", "N/A")
    revisions = result.get("revision_count", 0)
    logger.info("Sources retrieved: %d", len(sources))
    logger.info("Critic score:      %s/10", score)
    logger.info("Revision cycles:   %d", revisions)


if __name__ == "__main__":
    main()
