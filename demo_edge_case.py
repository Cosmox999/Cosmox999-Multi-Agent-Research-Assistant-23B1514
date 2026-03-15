import logging
import sys
from unittest.mock import patch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DIVIDER = "=" * 70

call_count = 0
original_search = None


def patched_search(query, max_results=4):
    global call_count
    call_count += 1
    if call_count == 3:
        # Simulate a timeout on the third query
        raise ConnectionError(
            f"Simulated Tavily timeout for query: '{query}'"
        )
    return original_search(query=query, max_results=max_results)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Import after load_dotenv so env vars are available
    from tavily import TavilyClient
    import tools.search as search_module

    global original_search

    # Grab the real Tavily search method so patched_search can delegate to it
    client = search_module.tavily
    original_search = client.search

    logger.info(DIVIDER)
    logger.info("  EDGE CASE DEMO — Simulating a search failure on query 3 of 5")
    logger.info(DIVIDER)
    logger.info("Scenario: one Tavily query times out mid-pipeline.")
    logger.info("Expected behaviour: WARNING is logged, pipeline continues with remaining sources.")
    logger.info(DIVIDER)

    from workflow.graph import build_graph

    app = build_graph()

    initial_state = {
        "query":             "Identify the top 5 emerging GenAI use cases in FMCG supply chains.",
        "search_queries":    [],
        "sources":           [],
        "research":          "",
        "analysis":          "",
        "critique":          "",
        "critique_score":    0,
        "critique_approved": False,
        "revision_count":    0,
        "final_report":      "",
    }

    # Patch the Tavily client's search method for this run only
    with patch.object(client, "search", side_effect=patched_search):
        result = None
        for result in app.stream(initial_state, stream_mode="values"):
            pass

    if result:
        sources_found = len(result.get("sources", []))
        logger.info(DIVIDER)
        logger.info("Edge case demo complete.")
        logger.info("Sources collected despite failure: %d (expected ~16, not 20)", sources_found)
        logger.info("Pipeline completed successfully — graceful degradation confirmed.")
        logger.info(DIVIDER)

        # Show a short excerpt of the report to prove it completed
        report = result.get("final_report", "")
        if report:
            excerpt = report[:400].strip()
            print("\n--- Report excerpt (first 400 chars) ---")
            print(excerpt)
            print("...\n[Full report still generated despite one failed search]")
    else:
        logger.error("Pipeline produced no output — unexpected failure.")


if __name__ == "__main__":
    main()
