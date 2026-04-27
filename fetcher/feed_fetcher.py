import feedparser
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup
import trafilatura
from typing import Dict, List
import json
import re

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
                    published_date = datetime.now(timezone.utc)

                if published_date > self.since_date:
                    if 'tags' in entry.keys():
                        category = entry['tags'][0].term.lower()
                    else:
                        category = 'NA'
                    rss_urls.append({'link':entry['link'],'category':category})

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
                    print("Unable to find the published date")

                if category in metadata['fetch_type']:
                    if published_date is not None and published_date > self.since_date:
                        no_rss_urls.append({'link':link,'category':category.lower()})

            if no_rss_urls != []:
                self.scrapped_links[source] = no_rss_urls


    def get_scrapped_links(self):
        print(self.scrapped_links)

    def extract_category(self):
        for source, articles in self.scrapped_links.items():
            for article in articles:
                if article['category'] == 'NA':
                    downloaded = trafilatura.fetch_url(article['link'])
                    article_metadata = trafilatura.extract_metadata(downloaded)
                    if article_metadata.categories != []:
                        article['category'] = article_metadata.categories[0]

    def extract_feed(self):
        scraped_feed ={}
        for source, articles in self.scrapped_links.items():
            for article in articles:
                category = article['category'].lower()
                if category in ['product','model','research']:
                    loader = trafilatura.fetch_url(article['link'])
                    content = trafilatura.extract(loader)
                    metadata = trafilatura.extract_metadata(loader)

                    if category not in scraped_feed:
                        scraped_feed[category] = []

                    scraped_feed[category].append({'title':metadata.title,'source':article['link'],'content':content})
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
