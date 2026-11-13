import json
from json import JSONDecodeError
from dotenv import load_dotenv
from logging_setup import setup_logger
from processing.article_filter import ArticleFilter
from fetcher.exa_fetcher import fetch_exa_articles
from groq import Groq
from groq_retry import run_groq_with_retry
from prompts import article_summary_prompt
load_dotenv()
from pathlib import Path
import asyncio

filtered_articles_path = Path(__file__).resolve().parent.parent / 'filtered_articles.json'

# Setting up the summarizer logger
summarize_logger = setup_logger('summarize_logger')

#Filtering articles
def fetch_and_filter():
    articles = fetch_exa_articles()
    article_filter = ArticleFilter(articles)
    filtered = article_filter.filter_articles(target_count=12)
    article_filter.save_filtered_articles(filtered)

def load_filtered_articles():
    try:
        with open(filtered_articles_path,'r',encoding='utf-8') as f:
            filtered_articles = json.load(f)

    except FileNotFoundError as e:
        return e
    except JSONDecodeError:
        summarize_logger.error(f"Unable to decode JSON file: {filtered_articles_path}")
    else:
        summarize_logger.info(f"Loaded {len(filtered_articles.values())} filtered articles")
        return filtered_articles

client = Groq()
def summarize_with_retry(content,category):
    system_prompt = article_summary_prompt(category)
    def make_call():
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {'role': 'system', 'content': system_prompt},
                {"role": 'user', 'content': content}],
        )
        return completion.choices[0].message.content
    return run_groq_with_retry(make_call)


async def summarize(refresh: bool = True):
    if refresh:
        await asyncio.to_thread(fetch_and_filter)
        result = load_filtered_articles()

    else:
        result = load_filtered_articles()
        if isinstance(result, FileNotFoundError):
            await asyncio.to_thread(fetch_and_filter)
            result = load_filtered_articles()

    for category, articles in result.items():
        category_summaries = []
        for article in articles:
            summary = await asyncio.to_thread(summarize_with_retry,article['content'],category)
            summarized_article = {
                "title": article['title'],
                "summary": summary,
                "source": article['url']
            }
            category_summaries.append(summarized_article)
            summarize_logger.info(f"Summarized the news from {article['url']} ")

        yield category, category_summaries
    summarize_logger.info("All the weekly news are summarized..")

