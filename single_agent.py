"""
=============================================================================
 SINGLE AGENT WITH TOOLS  (LangChain, Mistral via Ollama)
=============================================================================

This is the LangChain agent pattern (like the OpenAI example you found),
but adapted to:
   - run on your LOCAL Mistral (Ollama) instead of paid OpenAI
   - use YOUR project's real tools: web search + page extraction
   - COUNT TOKENS on every model call (needed for the research)

The agent decides ON ITS OWN when to call a tool. That is what makes it a
real "agent with tools" and not just a chatbot.
=============================================================================
"""

import os
import requests
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# 1) TOOLS  (these are your project's tools, reused directly)
# -----------------------------------------------------------------------------
SEARCH_API_KEY = os.getenv("SERP_API_KEY", "")  # same key your project uses

@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information about a topic.
    Use this when the question needs facts, news, or anything you don't
    already know. Returns a list of result titles, urls and snippets."""
    try:
        r = requests.get(
            "https://www.searchapi.io/api/v1/search",
            params={"q": query, "engine": "google", "api_key": SEARCH_API_KEY},
            timeout=10,
        )
        data = r.json()
        out = []
        for item in data.get("organic_results", [])[:5]:
            out.append(f"- {item.get('title')} ({item.get('link')}): "
                       f"{item.get('snippet')}")
        return "\n".join(out) if out else "No results found."
    except Exception as e:
        return f"SEARCH_ERROR: {e}"


@tool
def extract_page(url: str) -> str:
    """Read the main text content of a web page given its URL.
    Use this after search_web when you need the full details from a page."""
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


TOOLS = [search_web, extract_page]

# -----------------------------------------------------------------------------
# 2) THE MODEL  (local Mistral via Ollama)
# -----------------------------------------------------------------------------
llm = ChatOllama(model="mistral", temperature=0.0)

# -----------------------------------------------------------------------------
# 3) THE AGENT  (model + tools + instructions, runs the tool loop for us)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a research assistant with web tools.
- Use search_web to find information.
- Use extract_page to read a specific result in detail.
- Base your answer only on what the tools return; do not make things up.
- Give a clear, concise final answer."""

single_agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

# -----------------------------------------------------------------------------
# 4) RUN IT + COUNT TOKENS
# -----------------------------------------------------------------------------
def run_single_agent(question):
    """Run the single agent on a question. Returns (answer, total_tokens)."""
    result = single_agent.invoke({"messages": [HumanMessage(content=question)]})

    # the full message list includes every model + tool step
    messages = result["messages"]

    # final answer is the last message's content
    answer = messages[-1].content

    # sum tokens across every AI message that reported usage
    total_tokens = 0
    for m in messages:
        um = getattr(m, "usage_metadata", None)
        if um:
            total_tokens += um.get("total_tokens", 0)

    return answer, total_tokens


if __name__ == "__main__":
    q = "Who won the most recent FIFA World Cup and in year 2022?"
    print(f"Question: {q}\n")
    answer, tokens = run_single_agent(q)
    print(f"\nAnswer: {answer}")
    print(f"Total tokens used: {tokens}")
