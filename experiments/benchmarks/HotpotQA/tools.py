"""Reusable tools for HotpotQA benchmark agents."""

from __future__ import annotations

import re

from langchain_core.tools import tool


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a topic and return a summary.

    Use this to look up factual information about people, places, events,
    or concepts. Returns the first few sentences of the matching article.
    Examples: "Albert Einstein", "Battle of Gettysburg", "Python programming".
    """
    try:
        import wikipedia

        results = wikipedia.search(query, results=3)
        if not results:
            return f"No Wikipedia results for '{query}'."
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
            summary = page.summary
            if len(summary) > 1500:
                summary = summary[:1500].rsplit(".", 1)[0] + "."
            return summary
        except wikipedia.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                summary = page.summary
                if len(summary) > 1500:
                    summary = summary[:1500].rsplit(".", 1)[0] + "."
                return summary
            except Exception:
                return f"Disambiguation: possible matches are {e.options[:5]}"
        except wikipedia.PageError:
            return f"No Wikipedia page found for '{results[0]}'."
    except Exception as exc:
        return f"Error: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the web for information and return top results.

    Use this for recent events, lesser-known facts, or when Wikipedia
    doesn't have the answer. Returns titles and snippets from top results.
    Examples: "Who directed the 2023 film Oppenheimer?", "population of Tokyo 2024".
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No web results for '{query}'."
        parts = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            parts.append(f"- {title}: {body}")
        return "\n".join(parts)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def lookup_keyword(keyword: str, context: str) -> str:
    """Search within provided context paragraphs for sentences containing a keyword.

    Use this to find specific facts within the HotpotQA context without reading
    everything. Returns all sentences that mention the keyword (case-insensitive).
    Examples: keyword="founded", context="<the context text>"
    """
    keyword_lower = keyword.lower()
    sentences = re.split(r"(?<=[.!?])\s+", context)
    matches = [s.strip() for s in sentences if keyword_lower in s.lower()]
    if not matches:
        return f"No sentences containing '{keyword}' found in context."
    return "\n".join(matches)


TOOLS = [wikipedia_search, web_search, lookup_keyword]
