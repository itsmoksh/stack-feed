# Processing Component - Stack Feed

The processing component turns the raw weekly feed into a categorized, summarized digest. It sits between the fetchers and the Discord/RAG components.

## Overview

The component has two stages:

1. **Article Categorizer** - assigns fetched company articles to semantic categories and removes categories that are not included in the digest.
2. **Summarizer** - combines categorized articles with Gmail newsletters and summarizes every item into 3 to 5 bullet points.

The processed article content is stored in `latest_news.json`. This file is used both for summarization and for ingestion into the RAG pipeline.

## Processing Flow

```text
FeedFetcher
    |
    v
Raw articles grouped by source
    |
    v
ArticleCategorizer
    |-- build category anchors
    |-- semantically chunk article content
    |-- score chunks and title against every anchor
    |-- assign a category
    |-- keep digest categories
    |
    v
Categorized articles + Gmail newsletters
    |
    v
latest_news.json
    |
    +--> Groq summaries for the Discord digest
    |
    +--> Qdrant ingestion for RAG
```

## Components

### 1. Article Categorizer

**File:** `article_categorizer.py`

**Anchor configuration:** `anchor_config.json`

The categorizer uses `sentence-transformers/all-MiniLM-L6-v2` to compare each article with five configured category anchors:

- `Model Release`
- `Safety`
- `Product/Tooling`
- `Business/Company`
- `Research`

Each anchor contains example sentences that describe the type of article expected in that category. The embeddings of those examples are averaged and normalized to create one centroid per category.

#### Working

1. Builds a normalized embedding centroid for every category in `anchor_config.json`.
2. Chunks each article by semantic similarity with Chonkie's `SemanticChunker`.
3. Embeds every chunk and the article title with `all-MiniLM-L6-v2`.
4. Computes cosine similarity between each chunk and each category centroid.
5. Averages the chunk similarities to produce one body score per category.
6. Computes a separate title score for every category.
7. Assigns the article using the following rules:

| Condition | Result |
|---|---|
| Best body score is below `0.30` | `Uncategorized` |
| Difference between the top two body scores is greater than `0.03` | Best body category |
| Body scores are close, but the top two title scores differ by more than `0.02` | Best title category |
| Both body and title scores are close | `General` |

The public digest keeps `Model Release`, `Safety`, `Research`, `Product/Tooling`, and `General`. Articles classified as `Business/Company` or `Uncategorized` are currently excluded by `select_articles()`.

The selected output is grouped by category:

```json
{
  "Research": [
    {
      "title": "Article title",
      "content": "Extracted article text",
      "url": "https://example.com/article"
    }
  ]
}
```

### 2. Summarizer

**File:** `summarizer.py`

The summarizer coordinates fetching, categorization, persistence, and LLM summarization.

#### Working

1. Loads source settings from `fetcher/config.json` and category examples from `anchor_config.json`.
2. Uses `FeedFetcher` to collect articles published during the last seven days.
3. Passes company-feed articles through `ArticleCategorizer`.
4. Fetches configured Gmail newsletters separately and adds them under the `newsletter` key without semantic categorization.
5. Writes the categorized raw content to the project-root `latest_news.json`.
6. Sends each item to Groq's `openai/gpt-oss-120b` model with its category as context.
7. Returns 3 to 5 concise bullet points per item while preserving its title and source.

Groq rate-limit responses are retried up to five times. When the API supplies a retry delay, the summarizer waits for that duration before trying again.



