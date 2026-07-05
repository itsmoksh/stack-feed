# Fetcher Component - Stack Feed

The fetcher is the first component of Stack Feed. It automatically fetches the latest weekly articles, blogs, and newsletters from RSS feeds, non-RSS pages, and Gmail. It serves as the data collection layer that gathers raw content before company-feed articles are categorized and all selected items are summarized.

## Overview

There are two fetchers:

1. **Gmail Fetcher** - extracts newsletters from your Gmail inbox.
2. **Feed Fetcher** - collects articles from AI company RSS feeds and custom pages.

`config.json` stores the source URLs, non-RSS selectors, and Gmail senders. `FeedFetcher` returns articles grouped by source; the processing component categorizes them, adds newsletters, and stores the result in `latest_news.json`.

---

## Architecture

![Fetcher Architecture](../assets/fetcher_workflow.png)

## Components

### 1. Gmail Fetcher

**File:** `gmail_fetcher.py`

Fetches newsletters directly from your Gmail inbox using the Gmail API.

#### Set up Gmail credentials

- Go to the [Google Cloud Console](https://console.cloud.google.com/).
- Select an existing project or create a new one.
- Go to APIs and Services, find the Gmail API in the library, and enable it.
- Go to Credentials in the left sidebar and configure the consent screen if needed.
  - Enter the app name and user support email under **App Information**, then click Next.
  - Choose an internal or external audience.
  - Enter your email address in the contact information.
  - Click Finish and Create.
- Create an OAuth client and choose **Web Application** as the application type.
- Enter a name and click Create.
- Download the OAuth client JSON and save it as `fetcher/gmail_credentials.json`.

#### Working

1. Authenticates with the Gmail API using OAuth credentials and creates or refreshes a token.
2. Queries the inbox for messages from each configured sender during the last seven days.
3. Extracts and cleans plain-text or HTML email content.
4. Returns each newsletter with its title, content, and sender address.

---

### 2. Feed Fetcher

**File:** `feed_fetcher.py`

Collects articles directly from AI company websites.

#### Supported Sources

**RSS sources:** fetches weekly article URLs with Feedparser.

- OpenAI
- DeepMind
- Mistral AI

**Non-RSS sources:** extracts weekly article URLs with Beautiful Soup.

- Anthropic Newsroom

#### Working

1. Fetches entries from the configured RSS feeds.
2. Extracts article URLs and dates from configured non-RSS pages.
3. Keeps articles published during the last seven days.
4. Extracts each article's title and content with Trafilatura.
5. Returns articles grouped by source with `title`, `url`, and `content`.

The feed fetcher does not assign categories. `processing/article_categorizer.py` performs semantic categorization after fetching. See the [Processing README](../processing/README.md) for the classification rules.
