import json
from fetcher.gmail_fetcher import GmailFetcher
from fetcher.feed_fetcher import FeedFetcher
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq()
def extract_news():
    with open('fetcher/config.json','r') as f:
        configs = json.load(f)


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
    latest_news.update({'newsletter':gmail_news})

    #Creating a Json
    with open('latest_news.json','w') as f:
        json.dump(latest_news,f)



system_prompt = """You are StackFeed, an AI digest assistant for a Discord community 
of AI engineering learners and enthusiasts.

Your job is to summarize AI company blog posts into clear, 
insightful bullet points that help learners quickly understand 
what was released, why it matters.

Follow these rules strictly:
- Summarize in exactly 3 to 5 bullet points
- Each bullet point must be one clear, complete sentence
- Start each bullet with a strong action word like "Released", 
  "Introduced", "Improved", "Announced", "Launched"
- Focus on: what it is, what it does, why it matters to developers
- Use simple language — avoid unnecessary jargon
- If a benchmark or metric is mentioned, include it — numbers matter
- Never add information not present in the article
- Never use phrases like "The article says" or "According to"
- Never invent comparisons or context from outside the article
- Do not make up things, stick to  the article only and always end with one bullet on practical impact for developers if possible.
"""

def summarize():
    try:
        with open('latest_news.json','r') as f:
            latest_news = json.load(f)
    except FileNotFoundError:
        extract_news()
        with open('latest_news.json','r') as f:
            latest_news = json.load(f)

    print('Summarizing news...')
    for category, articles in latest_news.items():
        print(category.capitalize(),"\n")
        for article in articles:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{'role':'system','content':system_prompt},{"role": "user", "content":article['content']}]
            )
            print(article['title'])
            print(completion.choices[0].message.content)
            print(article['source'],'\n')

if __name__ == '__main__':
    summarize()
