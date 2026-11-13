import re
import json
from pathlib import Path
from dotenv import load_dotenv
from groq_retry import run_groq_with_retry
from groq import Groq
from prompts import TABLE_REFORMAT_SYSTEM_PROMPT, table_reformat_prompt
load_dotenv()

filtered_articles_path = Path(__file__).resolve().parent.parent / 'filtered_articles.json'

groq_client = Groq()

def extract_markdown_tables(text: str) -> list[dict]:
    lines = text.split("\n")
    tables = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
                header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
                row_start = i
                i += 2
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    rows.append(cells)
                    i += 1
                tables.append({
                    "start_line": row_start,
                    "end_line": i,
                    "headers": header_cells,
                    "rows": rows,
                    "raw_markdown": "\n".join(lines[row_start:i]),  # <-- new: keep original text
                })
                continue
        i += 1

    return tables

def table_to_sentences_llm(table_markdown: str, article_title: str) -> str:
    print(f"Going this markdown: {table_markdown}")
    prompt = table_reformat_prompt(article_title, table_markdown)

    def make_call():
        return groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
            {
                "role": "system",
                "content": TABLE_REFORMAT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt}
            ],
            include_reasoning=False,
            max_completion_tokens=4096,
            reasoning_effort="low",
        )
    response = run_groq_with_retry(make_call)
    message = response.choices[0].message.content
    return message.strip()



def preprocess_article_tables(article_text: str, article_title: str) -> str:
    tables = extract_markdown_tables(article_text)
    if not tables:
        return article_text

    lines = article_text.split("\n")
    for table in reversed(tables):
        sentences = table_to_sentences_llm(table["raw_markdown"], article_title)

        lines[table["start_line"]:table["end_line"]] = [sentences]

    return "\n".join(lines)

if __name__ == "__main__":
    with open(filtered_articles_path, 'r', encoding='utf-8') as f:
        filtered_articles = json.load(f)

    for category, articles in filtered_articles.items():
        print("="*20, "Processing: ", category, "="*20)
        for article in articles:
            print("Title: ", article['title'])
            preprocessed_article = preprocess_article_tables(article['content'], article['title'])
            print(preprocessed_article)
            break



