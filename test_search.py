import os, requests
from dotenv import load_dotenv
load_dotenv()


key = os.getenv("SERP_API_KEY", "")
print("Key loaded?", "YES" if key else "NO — key is empty!")

r = requests.get(
    "https://www.searchapi.io/api/v1/search",
    params={"q": "most recent FIFA World Cup winner",
            "engine": "google", "api_key": key},
    timeout=10,
)
print("HTTP status:", r.status_code)
data = r.json()
print("Top-level keys returned:", list(data.keys()))
print("Number of organic results:", len(data.get("organic_results", [])))
if "error" in data:
    print("API ERROR MESSAGE:", data["error"])
if data.get("organic_results"):
    print("First result:", data["organic_results"][0].get("title"))