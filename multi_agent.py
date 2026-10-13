"""
=============================================================================
 MULTI-AGENT SYSTEM  (Manager + Sub-agents, each with a tool)
 LangChain / Mistral via Ollama
=============================================================================

Architecture
------------
        User query
            |
        [ MANAGER ]  <- Mistral reads the query and picks ONE sub-agent
            |
   ---------+-----------------
   |            |            |
[SEARCH]     [SCRAPE]     [ANSWER]     <- sub-agents
 +web         +web         (no tool,
  search       scraper      just the model)

Why routing is done this way
----------------------------
Local Mistral is UNRELIABLE at emitting structured tool-calls (we saw the
single agent print tool calls as text instead of running them). So instead
of trusting the model to emit tool-call JSON, the MANAGER only has to output
ONE WORD (SEARCH / SCRAPE / ANSWER) -- something Mistral does reliably -- and
the CODE guarantees the chosen tool actually runs. This is a real
manager -> sub-agent design that is honest about the local model's limits.

Every model call is token-counted so this can be compared against the single
agent AT EQUAL COMPUTE (the research question).
=============================================================================
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
llm = ChatOllama(model="mistral", temperature=0.0)

SEARCH_API_KEY = os.getenv("SERP_API_KEY", "")


# -----------------------------------------------------------------------------
# TOKEN-COUNTED MODEL CALL  (every agent uses this)
# -----------------------------------------------------------------------------
def ask(system_prompt, user_prompt):
    """One model call. Returns (text, tokens)."""
    resp = llm.invoke([SystemMessage(content=system_prompt),
                       HumanMessage(content=user_prompt)])
    tokens = 0
    if getattr(resp, "usage_metadata", None):
        tokens = resp.usage_metadata.get("total_tokens", 0)
    return resp.content.strip(), tokens


# -----------------------------------------------------------------------------
# THE TOOLS  (your project's real tools)
# -----------------------------------------------------------------------------
def tool_search_web(query):
    """Google search via SearchAPI. Returns formatted results text."""
    try:
        r = requests.get(
            "https://www.searchapi.io/api/v1/search",
            params={"q": query, "engine": "google", "api_key": SEARCH_API_KEY},
            timeout=10,
        )
        data = r.json()
        out, first_url = [], None
        for item in data.get("organic_results", [])[:5]:
            if first_url is None:
                first_url = item.get("link")
            out.append(f"- {item.get('title')} ({item.get('link')}): "
                       f"{item.get('snippet')}")
        return ("\n".join(out) if out else "No results found."), first_url
    except Exception as e:
        return f"SEARCH_ERROR: {e}", None


def tool_extract_page(url):
    """Scrape main text from a page."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = [l.strip() for l in soup.get_text("\n").splitlines()]
        lines = [l for l in lines if len(l) > 30]
        return "\n".join(lines)[:3000]
    except Exception as e:
        return f"EXTRACT_ERROR: {e}"


# -----------------------------------------------------------------------------
# THE MANAGER AGENT  (decides which sub-agent handles the query)
# -----------------------------------------------------------------------------
MANAGER_SYS = """You are a manager that routes a user's question to ONE worker.
Reply with EXACTLY ONE WORD, nothing else:
- SEARCH  : if the question needs current facts, news, or web information
- SCRAPE  : if the question contains a specific URL to read
- ANSWER  : if it is general knowledge you can answer without the web
Your reply must be only one word: SEARCH, SCRAPE, or ANSWER."""

def manager_route(query):
    decision, tokens = ask(MANAGER_SYS, f"Question: {query}")
    d = decision.upper()
    if "SCRAPE" in d or "http" in query:
        route = "SCRAPE"
    elif "SEARCH" in d:
        route = "SEARCH"
    else:
        route = "ANSWER"
    return route, tokens


# -----------------------------------------------------------------------------
# SUB-AGENTS  (each owns a tool and phrases the final answer)
# -----------------------------------------------------------------------------
def search_agent(query):
    """Sub-agent 1: uses the web-search tool, then answers from results."""
    results, _ = tool_search_web(query)
    ans, tokens = ask(
        "You are a research sub-agent. Answer the question using ONLY the "
        "search results provided. Be concise and factual.",
        f"Question: {query}\n\nSearch results:\n{results}",
    )
    return ans, tokens, results


def scrape_agent(query):
    """Sub-agent 2: extracts a URL from the query, scrapes it, then answers."""
    urls = re.findall(r"https?://\S+", query)
    if not urls:
        return "No URL found to scrape.", 0, ""
    content = tool_extract_page(urls[0])
    ans, tokens = ask(
        "You are a reading sub-agent. Answer the question using ONLY the page "
        "content provided. Be concise.",
        f"Question: {query}\n\nPage content:\n{content}",
    )
    return ans, tokens, content[:300]


def answer_agent(query):
    """Sub-agent 3: no tool, answers from the model's own knowledge."""
    ans, tokens = ask(
        "You are a general-knowledge sub-agent. Answer concisely.",
        f"Question: {query}",
    )
    return ans, tokens, ""


# -----------------------------------------------------------------------------
# THE FULL MULTI-AGENT RUN  (manager -> chosen sub-agent), token-counted
# -----------------------------------------------------------------------------
def run_multi_agent(query, verbose=True):
    total_tokens = 0

    # 1) Manager decides the route
    route, t_manager = manager_route(query)
    total_tokens += t_manager
    if verbose:
        print(f"  [Manager] routed to: {route}  (tokens={t_manager})")

    # 2) Chosen sub-agent runs (its tool actually executes)
    if route == "SEARCH":
        answer, t_sub, evidence = search_agent(query)
    elif route == "SCRAPE":
        answer, t_sub, evidence = scrape_agent(query)
    else:
        answer, t_sub, evidence = answer_agent(query)
    total_tokens += t_sub

    if verbose:
        print(f"  [{route} sub-agent] answered  (tokens={t_sub})")

    return {
        "answer": answer,
        "route": route,
        "tokens": total_tokens,
        "evidence": evidence,
    }


if __name__ == "__main__":
    tests = [
        "Who won the most recent FIFA World Cup and in what year?",
        "What is the capital of France?",
        "Summarize this page: https://en.wikipedia.org/wiki/FIFA_World_Cup",
    ]
    for q in tests:
        print(f"\nQuestion: {q}")
        result = run_multi_agent(q)
        print(f"  Answer : {result['answer']}")
        print(f"  Route  : {result['route']}")
        print(f"  Tokens : {result['tokens']}")
