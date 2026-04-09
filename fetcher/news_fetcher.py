import feedparser
from datetime import datetime, timezone, timedelta
import trafilatura

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

    rss_blogs = {}
    for link in feed_links:
        loader = trafilatura.fetch_url(link)
        content = trafilatura.extract(loader)
        metadata = trafilatura.extract_metadata(loader)
        rss_blogs[metadata.title] = {'content':content, 'source':metadata.url}

    return rss_blogs

since_date = datetime.now() - timedelta(days=7)
articles = extract_news('https://test/blog/feed.xml',since_date) #rss feed url here

if __name__ == '__main__':
    # Checking working of it.
    for title, news in articles.items():
        print(title)
        print(news['content'])
        print(news['source'])
