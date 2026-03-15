import os
import logging
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_multiple(queries: list, max_per_query: int = 4) -> tuple:
    """
    Run multiple Tavily searches and return:
      sources  — deduplicated list of {title, url, content}
      combined — single string of all content for LLM consumption
    """
    seen_urls = set()
    sources = []

    for query in queries:
        logger.info("[SEARCH] Query: %s", query)
        try:
            results = tavily.search(query=query, max_results=max_per_query)
            for r in results.get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "title":   r.get("title", "Untitled"),
                        "url":     url,
                        "content": r.get("content", ""),
                    })
                    logger.debug("[SEARCH] Found: %s", url)
        except Exception as exc:
            logger.warning("[SEARCH] Failed for query '%s': %s", query, exc)

    logger.info("[SEARCH] Total unique sources collected: %d", len(sources))

    # Build a single text block the researcher LLM will summarise.
    # Include title + URL so later agents can cite real links.
    combined_parts = []
    for s in sources:
        combined_parts.append(
            f"SOURCE: {s['title']}\nURL: {s['url']}\n{s['content']}"
        )
    combined = "\n\n---\n\n".join(combined_parts)

    return sources, combined
