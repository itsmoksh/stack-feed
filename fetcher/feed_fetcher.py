import feedparser
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup
import trafilatura
from typing import Dict
import json
from pathlib import Path
from logging_setup import setup_logger

#settitng up the loger
log_path = Path(__file__).parent.parent/'stack_feed.log'
feed_logger = setup_logger("feed_logger",log_path)

class FeedFetcher:
    def __init__(self,config:Dict):
        self.since_date = datetime.today() - timedelta(days=7)
        self.rss_sources = config.get('rss_sources')
        self.no_rss_sources = config.get('no_rss_sources')
        self.scrapped_links= {}

    def extract_rss_urls(self):
        for source,metadata in self.rss_sources.items():
            rss_urls = []
            feed = feedparser.parse(metadata['url'])
            for entry in feed['entries']:
                try:
                    published_date = datetime(*entry['published_parsed'][:6])
                except:
                    feed_logger.debug(f"No published date found for {entry['link']}, Skipping..")
                    continue

                if published_date > self.since_date:
                    rss_urls.append(entry['link'])
                    feed_logger.info(f"RSS URL found from {source}; URL: {entry['link']}, Published Date: {published_date}")

            if rss_urls:
                self.scrapped_links[source] = rss_urls


    def extract_no_rss_urls(self):
        for source,metadata in self.no_rss_sources.items():
            no_rss_urls = []
            response = requests.get(metadata['url'])
            soup = BeautifulSoup(response.content, 'html.parser')

            articles = soup.find_all('a', class_=metadata['class'])
            for article in articles:
                link = article['href']
                url, published_date = None, None
                try:
                    _,add = link.split('/news')
                    url = f'{metadata["url"]}{add}'
                    time_tag = article.find('time')
                    date_str = time_tag.get_text()
                    published_date = datetime.strptime(date_str, "%b %d, %Y")

                except (ValueError, AttributeError) as e:
                    feed_logger.error(f"Unable to get the URL or published date of {link}, got {e}")

                if url and published_date:
                    if published_date > self.since_date:
                        no_rss_urls.append(url)
                        feed_logger.info(f"Non RSS URL found from {source}; URL: {url}, Published Date: {published_date}")

            if no_rss_urls:
                self.scrapped_links[source] = no_rss_urls


    def get_scrapped_links(self):
        print(self.scrapped_links)


    def extract_feed(self):
        scraped_feed ={}
        self.extract_rss_urls()
        self.extract_no_rss_urls()
        for source, links in self.scrapped_links.items():
            scraped_feed[source] = []
            for link in links:
                loader = trafilatura.fetch_url(link)
                if loader is None:
                    feed_logger.warning(f"Failed to fetch {link}, skipping")
                    continue
                content = trafilatura.extract(loader)
                metadata = trafilatura.extract_metadata(loader)
                title = metadata.title if metadata else None
                feed_logger.info(f"Extracted content from {source}; Title: {title}, URL: {link}")
                scraped_feed[source].append({'title': title, 'url': link, 'content': content})

        return scraped_feed

if __name__ == '__main__':
    with open('config.json','r') as f:
        config = json.load(f)

    f = FeedFetcher(config)
    for source,articles in f.extract_feed().items():
        print(source)
        for article in articles:
            print(f"{article['title']}\n{article['content']}\nSource: {article['url']}\n")

