import feedparser
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup
import trafilatura
from typing import Dict
import json
import re
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
                    feed_logger.debug(f"No published date found for {entry['link']}, setting to current:{datetime.now(timezone.utc)}")
                    published_date = datetime.now(timezone.utc)

                if published_date > self.since_date:
                    if 'tags' in entry and entry['tags']:
                        category = entry['tags'][0].term.lower()
                    else:
                        category = 'NA'
                    rss_urls.append({'link':entry['link'],'category':category})
                    feed_logger.info(f"RSS URL found, URL: {entry['link']}, Category: {category}, Published Date: {published_date}")

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
                text = article.get_text()

                _,add = link.split('/news')
                url = f'{metadata["url"]}{add}'

                pattern = r"(\w+\s+\d{1,2},\s+\d{4})([A-Z][a-z]+)"
                match = re.search(pattern,text)
                date_str,category = match.groups()
                published_date = None
                try:
                    published_date = datetime.strptime(date_str, "%b %d, %Y")
                except:
                    feed_logger.debug(f"No published date found for {link}")

                if category in metadata['fetch_type']:
                    if published_date is not None and published_date > self.since_date:
                        no_rss_urls.append({'link':link,'category':category.lower()})
                        feed_logger.info(f"Non RSS URL found; URL: {link}, Category: {category}, Published Date: {published_date}")

            if no_rss_urls:
                self.scrapped_links[source] = no_rss_urls


    def get_scrapped_links(self):
        print(self.scrapped_links)

    def extract_category(self):
        for source, articles in self.scrapped_links.items():
            for article in articles:
                if article['category'] == 'NA':
                    try:
                        downloaded = trafilatura.fetch_url(article['link'])
                        article_metadata = trafilatura.extract_metadata(downloaded)
                        if article_metadata.categories != []:
                            article['category'] = article_metadata.categories[0]
                            feed_logger.info(f"Category found for {article['link']}, Category: {article_metadata.categories[0]}")
                        else:
                            feed_logger.debug(f"No category found for {article['link']}")

                    except Exception as e:
                        feed_logger.error(f"Failed to extract category for {article['link']}, got error: {e}")

    def extract_feed(self):
        scraped_feed ={}

        for source, articles in self.scrapped_links.items():
            for article in articles:
                category = article['category'].lower()
                if category in ['product','model','research']:
                    try:
                        loader = trafilatura.fetch_url(article['link'])
                        content = trafilatura.extract(loader)
                        metadata = trafilatura.extract_metadata(loader)

                        if category not in scraped_feed:
                            scraped_feed[category] = []

                        feed_logger.info(f"Extracted content from {metadata.title}, source: {article['link']}, category: {category}")
                        scraped_feed[category].append({'title': metadata.title, 'source': article['link'], 'content': content})

                    except Exception as e:
                        feed_logger.error(f"Unable to extract data from {article['link']}, got error: {e}")

        return scraped_feed

if __name__ == '__main__':
    with open('config.json','r') as f:
        config = json.load(f)

    feed = FeedFetcher(config)
    feed.extract_rss_urls()
    feed.extract_no_rss_urls()
    feed.extract_category()
    feed.get_scrapped_links()
    print(feed.extract_feed())
