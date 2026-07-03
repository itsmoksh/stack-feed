import json
from fetcher.feed_fetcher import FeedFetcher
import numpy as np
from sentence_transformers import SentenceTransformer
from chonkie import SemanticChunker
from logging_setup import setup_logger
from pathlib import Path

log_path = Path(__file__).parent.parent/'stack_feed.log'
categorize_logger = setup_logger('categorize_logger',log_path)

class ArticleCategorizer():
    def __init__(self, anchor_configs, feed):
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.anchors_embeddings = self._build_anchor_embeddings(anchor_configs)
        self.feed = feed

    def _build_anchor_embeddings(self, anchor_configs):
        cat_centroids = {}
        for category, sentences in anchor_configs.items():
            sentences_embed = self.embed_model.encode(sentences,normalize_embeddings=True)
            centroid = np.mean(sentences_embed, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            cat_centroids[category] = centroid
        categorize_logger.info("Successfully built anchor embeddings")
        return cat_centroids

    def similarity_chunker(self):
        chunks_per_article= {}
        chunker = SemanticChunker(
            embedding_model='all-MiniLM-L6-v2',
            threshold=0.8,  # Similarity threshold (0-1)
            chunk_size=2048,  # Maximum tokens per chunk
            similarity_window=3,  # Window for similarity calculation
            min_sentences_per_chunk=4
        )
        for source, articles in self.feed.items():
            for article in articles:
                chunk = chunker.chunk(article['content'])
                chunk_texts = [c.text for c in chunk]
                chunks_per_article[(article['title'], article['url'])] = chunk_texts
        categorize_logger.info("Chunked all the articles based on semantic similarity")
        return chunks_per_article

    def embed_chunks(self):
        embedded_articles = {}
        chunks_per_article = self.similarity_chunker()
        for article_info,chunks in chunks_per_article.items():
            embedded_chunks = self.embed_model.encode(chunks,normalize_embeddings=True)
            embedded_articles[article_info] = embedded_chunks
        categorize_logger.info("Embedded all the articles for finding the similarity score")
        return embedded_articles

    def categorize_articles(self, min_sim: float = 0.30, close_call_delta: float = 0.03, title_delta: float = 0.02):
        embedded_articles = self.embed_chunks()
        results = {}
        avg_diff_chunks_top2 = []
        avg_diff_titles_top2 = []
        for article_info, embedded_chunks in embedded_articles.items():
            title = article_info[0]
            title_embed = self.embed_model.encode(title, normalize_embeddings=True)

            chunk_scores = {}
            title_scores = {}
            for cat, centroid in self.anchors_embeddings.items():
                chunk_scores[cat] = np.mean([np.dot(chunk, centroid) for chunk in embedded_chunks])
                title_scores[cat] = np.dot(title_embed, centroid)

            #Chunks Top 2 scores diff
            sorted_chunk = sorted(chunk_scores.items(), key=lambda x: -x[1])
            top_cat, top_score = sorted_chunk[0]
            second_cat, second_score = sorted_chunk[1]
            avg_diff_chunks_top2.append(float(top_score - second_score))

            #Title Top 2 scores diff
            sorted_title = sorted(title_scores.items(), key=lambda x: -x[1])
            top_title_cat, top_title_score = sorted_title[0]
            second_title_cat, second_title_score = sorted_title[1]
            avg_diff_titles_top2.append(float(top_title_score - second_title_score))

            if top_score < min_sim:
                label = "Uncategorized"
                categorize_logger.debug(f"{article_info} is Uncategorized, because the Top Score: {top_score} < Minimum Sim: {min_sim}")

            elif (top_score - second_score) > close_call_delta:
                label = top_cat
                categorize_logger.info(f"{article_info} is categorized as {label} with the top score: {top_score}")

            else:
                if (top_title_score - second_title_score) > title_delta:
                    label = top_title_cat
                    categorize_logger.info(f"{article_info} talks about multiple categories, {label} is selected based on title score{top_title_score}")
                else:
                    label = "General"
                    categorize_logger.info(f"{article_info} talks about multiple categories, and have congested scores for both title and chunks. Hence labelled as {label}")

            results[article_info] = label
        categorize_logger.info(f"Differences between top 2 categories for chunks: {avg_diff_chunks_top2}")
        categorize_logger.info(f"Differences between top 2 categories for title: {avg_diff_titles_top2}")
        return results

    def select_articles(self):
        selected = {}
        include_categories = ["Model Release", "Safety", "Research", "Product/Tooling", "General"]
        categorized_articles = self.categorize_articles()
        url_with_content = {}
        for articles in self.feed.values():
            for article in articles:
                url_with_content[article['url']] = article['content']


        for article_info,label in categorized_articles.items():
            if label not in include_categories:
                continue
            title, url = article_info
            article_content = url_with_content.get(url)
            if label in selected:
                selected[label].append({
                    'title': title,
                    'content': article_content,
                    'url': url
                })
            else:
                selected[label] = [{'title': title,
                    'content': article_content,
                    'url': url}]
            categorize_logger.info(f"{article_info} is selected: {label}")

        return selected

if __name__ == "__main__":
    with open('anchor_config.json') as anchor_config:
        anchor_config = json.load(anchor_config)

    with open('../fetcher/config.json') as f:
        source_configs = json.load(f)

    fetcher= FeedFetcher(source_configs)
    feed = fetcher.extract_feed()

    c = ArticleCategorizer(anchor_config,feed)
    selected_articles = c.select_articles()
    for category, articles in selected_articles.items():
        print(category)
        for article in articles:
            print(f"Title: {article['title']}, URL: {article['url']}")

