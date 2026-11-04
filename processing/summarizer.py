import json
from json import JSONDecodeError
from groq import Groq
from dotenv import load_dotenv
from logging_setup import setup_logger
from processing.article_filter import ArticleFilter
from fetcher.exa_fetcher import fetch_exa_articles
from groq_retry import run_groq_with_retry
load_dotenv()
from pathlib import Path

filtered_articles_path = Path(__file__).resolve().parent.parent / 'filtered_articles.json'

#Setting up the summarizer logger
log_path = Path(__file__).parent.parent/'stack_feed.log'
summarize_logger = setup_logger('summarize_logger',log_path)

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
    system_prompt = f"""You are StackFeed, an AI digest assistant for a Discord community 
    of AI engineering learners and enthusiasts.

    Summarize the following {category} article into 3-5 clear bullet points 
    that help learners quickly understand what happened and why it matters.

    Rules:
    - Each bullet is one complete sentence, starting with a strong, accurate verb 
      matching what actually happened (a release, a finding, a safety measure, a deal — 
      don't force release-style language onto research or safety content). Do not bold verbs.
    - Include specific benchmarks or metrics if mentioned — numbers matter
    - Use simple language, avoid unnecessary jargon
    - Never add information not in the article, never invent comparisons, never say 
      "the article says" or "according to"
    - Include a practical-impact-for-developers bullet only if the article genuinely supports one
    - If the source is a newsletter, treat the assigned category only as a loose 
      hint, not a strict theme — newsletters bundle multiple unrelated news items together, 
      so summarize the distinct important stories it actually contains rather than forcing 
      everything into one category's framing

    Output format:
    - bullet 1
    - bullet 2
    - bullet 3
    - bullet 4 (if necessary)
    - bullet 5 (if necessary)"""
    def make_call():
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {'role': 'system', 'content': system_prompt},
                {"role": "user", "content": content}]
        )
        return completion.choices[0].message.content
    return run_groq_with_retry(make_call)


def summarize(refresh:bool = False):
    if refresh:
        fetch_and_filter()
        result = load_filtered_articles()

    else:
        result = load_filtered_articles()
        if isinstance(result, FileNotFoundError):
            fetch_and_filter()
            result = load_filtered_articles()

    summarized_news = {}
    for category, articles in result.items():
        summarized_news[category] = []
        for article in articles:
            summary = summarize_with_retry(article['content'],category)
            summarized_article={
                "title": article['title'],
                "summary": summary,
                "source": article['url']
            }
            summarized_news[category].append(summarized_article)
            summarize_logger.info(f"Summarized the news from {article['url']} ")
    summarize_logger.info("All the weekly news are summarized..")
    return summarized_news

if __name__ == '__main__':
    summarized_news = summarize(refresh=False)
    for category, articles in summarized_news.items():
        print(f"-----Category: {category}-------")
        for article in articles:
            print(article['title'])
            print(article['summary'])
            print(article['source'],'\n')

