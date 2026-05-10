import json
import time
from fetcher.gmail_fetcher import GmailFetcher
from fetcher.feed_fetcher import FeedFetcher
from groq import Groq
from dotenv import load_dotenv
from logging_setup import setup_logger
load_dotenv()

#Setting up the summarizer logger
log_path = 'stack_feed.log'
summarize_logger = setup_logger('summarize_logger',log_path)

client = Groq()
def extract_news():
    with open('fetcher/config.json','r') as f:
        configs = json.load(f)

    summarize_logger.info("Extracting latest weekly news from the sources")
    gmail_configs = configs.pop('gmail_sources')
    feed_configs = configs
    latest_news = {}

    # Getting latest news from rss and non-rss sources
    feed = FeedFetcher(feed_configs)
    feed.extract_rss_urls()
    feed.extract_no_rss_urls()
    feed.extract_category()
    feed_news = feed.extract_feed()
    latest_news.update(feed_news)

    # Getting newsletters
    gmail_news = []
    for _,email in gmail_configs.items():
        e_mail_news = GmailFetcher().fetch(email)
        gmail_news.extend(e_mail_news)
    if gmail_news:
        latest_news.update({'newsletter':gmail_news})

    #Creating a Json
    with open('latest_news.json','w') as f:
        json.dump(latest_news,f)
        summarize_logger.info("All the weekly news are stored in 'latest_news.json'")
    return latest_news

system_prompt = """You are StackFeed, an AI digest assistant for a Discord community 
of AI engineering learners and enthusiasts.

Your job is to summarize AI company blog posts into clear, 
insightful bullet points that help learners quickly understand 
what was released, why it matters.

Follow these rules strictly:
- Summarize in exactly 3 to 5 bullet points
- Each bullet point must be one clear, complete sentence
- Start each bullet with a strong action word like "Released", 
"Introduced", "Improved", "Announced", "Launched". Do not bold these. 
- Focus on: what it is, what it does, why it matters to developers
- Use simple language — avoid unnecessary jargon
- If a benchmark or metric is mentioned, include it — numbers matter
- Never add information not present in the article
- Never use phrases like "The article says" or "According to"
- Never invent comparisons or context from outside the article
- Do not make up things, stick to  the article only and always end with one bullet on practical impact for developers if possible.

Output format:
- bullet 1
- bullet 2
- bullet 3
- bullet 4 (if necessary)
- bullet 5 (if necessary)
"""

def summarize_article(content):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {"role": "user", "content": content}]
    )
    return completion.choices[0].message.content


def summarize(refresh = False,news_path = 'latest_news.json'):
    try:
        if refresh:
            latest_news = extract_news()
            with open(news_path,'w') as f:
                json.dump(latest_news,f)
        else:
            with open(news_path,'r') as f:
                latest_news = json.load(f)

    except FileNotFoundError:
        summarize_logger.info("No json files found for latest news, creating new ones")
        latest_news = extract_news()

    summarize_logger.info("Latest weekly news are loaded for summarization")

    summarized_news = {}
    for category, articles in latest_news.items():
        summarized_news[category] = []
        for article in articles:
            summary = summarize_article(article['content'])
            time.sleep(2) # To prevent the rate limit of Groq
            summarized_article={
                "title": article['title'],
                "summary": summary,
                "source": article['source']
            }
            summarized_news[category].append(summarized_article)
    summarize_logger.info("Weekly news are summarized..")
    return summarized_news

if __name__ == '__main__':
    summarized_news = summarize(refresh=True)
    for category, articles in summarized_news.items():
        print(category,'\n')
        for article in articles:
            print(article['title'],'\n')
            print(article['summary'],'\n')
            print(article['source'])

