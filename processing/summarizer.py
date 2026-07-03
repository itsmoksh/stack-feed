import json
import time
from fetcher.gmail_fetcher import GmailFetcher
from fetcher.feed_fetcher import FeedFetcher
from processing.article_categorizer import ArticleCategorizer
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from logging_setup import setup_logger
load_dotenv()
import re
from pathlib import Path

#Setting up the summarizer logger
log_path = Path(__file__).parent.parent/'stack_feed.log'
summarize_logger = setup_logger('summarize_logger',log_path)

#File paths
fetcher_config = Path(__file__).parent.parent/'fetcher/config.json'
anchor_config = Path(__file__).parent.parent/'processing/anchor_config.json'
news_path = Path(__file__).parent.parent/'latest_news.json'

client = Groq()
def extract_news():
    with open(fetcher_config, 'r') as f:
        fetcher_configs = json.load(f)
    with open(anchor_config, 'r') as f:
        anchor_configs = json.load(f)

    summarize_logger.info("Extracting latest weekly news from the sources")
    gmail_configs = fetcher_configs.pop('gmail_sources')
    latest_news = {}

    # Getting latest news from rss and non-rss sources and categorizing them
    feed_fetcher = FeedFetcher(fetcher_configs)
    feed = feed_fetcher.extract_feed()
    categorizer = ArticleCategorizer(anchor_configs,feed)
    final_feed = categorizer.select_articles()
    latest_news.update(final_feed)

    # Getting newsletters
    gmail_news = []
    for _,email in gmail_configs.items():
        e_mail_news = GmailFetcher().fetch(email)
        gmail_news.extend(e_mail_news)
    if gmail_news:
        latest_news.update({'newsletter':gmail_news})

    #Creating a Json
    with open(news_path, 'w') as f:
        json.dump(latest_news,f)
        summarize_logger.info("All the weekly news are stored in 'latest_news.json'")
    return latest_news

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
    for attempt in range(5):
        try:
            completion =  client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {"role": "user", "content": content}]
            )
            return completion.choices[0].message.content
        except RateLimitError as e:
            match = re.search(r"try again in ([\d.]+)(ms|s)",str(e))
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()
                wait_time = value / 1000 if unit == 'ms' else value
                summarize_logger.warning(f"Rate Limit Exceeded, retrying in {wait_time} seconds. Attempt #{attempt+1}")
                time.sleep(wait_time)
            else:
                time.sleep(60)
        raise RuntimeError(f"Exceed maximum number of attempts.")


def summarize(refresh:bool = False, news_path = news_path):

    if refresh:
        latest_news = extract_news()
    else:
        with open(news_path,'r') as f:
            latest_news = json.load(f)

    summarize_logger.info("Latest weekly news are loaded for summarization")

    summarized_news = {}
    for category, articles in latest_news.items():
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
    summarized_news = summarize(refresh=True)
    for category, articles in summarized_news.items():
        print(f"-----Category: {category}-------")
        for article in articles:
            print(article['title'])
            print(article['summary'])
            print(article['source'],'\n')

