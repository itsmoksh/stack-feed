from dotenv import load_dotenv
from pathlib import Path
from exa_py import Exa
from datetime import datetime, timedelta, timezone
import json
load_dotenv()
exa = Exa()

QUERIES = [
    {"query": "new AI model release or announcement", "category":"model release"},
    {"query": "new AI product or feature launch", "category":"product/tooling"},
    {"query": "AI safety and security update", "category":"safety"},
    {"query": "new research paper on language models or AI agents", "category":"research"}
]
 
def fetch_exa_articles():
    articles = []
    since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for q in QUERIES:
        try:
            result = exa.search(
                query = q['query'],
                category= "news",
                end_published_date=end,
                num_results=10,
                start_published_date=since,
                type="auto",
                contents={"text": True,"highlights":True}
            )
        except Exception as e:
            print(f"[exa] query failed: {q['query']!r}: {e}")
            continue

        for r in result.results:
            articles.append({
                "title":r.title,
                "content":r.text,
                "highlight": r.highlights,
                "url":r.url,
                "published": r.published_date,
                "category": q['category'],
            })
    return articles

if __name__ == "__main__":
    articles = fetch_exa_articles()
    articles_out = Path(__file__).resolve().parent.parent / "articles.json"
    with open(articles_out, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
