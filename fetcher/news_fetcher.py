import feedparser
from langchain_community.document_loaders import WebBaseLoader
from datetime import datetime, timezone, timedelta


def extract_news(rss_url: str, since_date: datetime):
    feeds = feedparser.parse(rss_url)
    feed_links = []
    for entry in feeds.entries:
        try:
            published_date = datetime(*entry['published_parsed'][:6])
        except:
            published_date = datetime.now(timezone.utc)

        if published_date > since_date:
            feed_links.append(entry['link'])

    loader_multiple_pages  = WebBaseLoader(feed_links)
    docs = loader_multiple_pages.load()
    articles = {}
    for doc in docs:
        articles[doc.metadata['title']] = [doc.page_content, doc.metadata.get('source','')]
    return articles


since_date = datetime.now() - timedelta(days=7)
news = extract_news('https://huggingface.co/blog/feed.xml',since_date)

for title, news in news.items():
    print(news[0])
    break