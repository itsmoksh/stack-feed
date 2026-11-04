import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse


import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typesafe_sdk import Choice, TypeSafeClient

from logging_setup import setup_logger

load_dotenv()

# Setup logger for filtering operations
log_path = Path(__file__).resolve().parent.parent / 'stack_feed.log'
filter_logger = setup_logger('filter_logger', log_path)

articles_path = Path(__file__).resolve().parent.parent / 'articles.json'
filtered_articles_path = Path(__file__).resolve().parent.parent / 'filtered_articles.json'
 

class ArticleFilter:
    """
    Filters raw articles fetched from Exa (or other sources) down to a curated set
    of ~10-12 high-quality, non-duplicate, technically relevant articles for AI engineers
    and data scientists.

    Filtering Pipeline:
    1. Length validation (article content > 500 characters, non-empty title and URL).
    2. URL deduplication (drops duplicate articles from overlapping search queries).
    3. Highlight semantic similarity deduplication (detects multi-publisher coverage of the same event).
    4. Relevance classification via TypeSafe AI (Jev model) using article highlights (excludes
       partnerships, corporate mergers, commercial tool integrations, and sales marketing).
    5. Capping and category balancing to yield ~10-12 articles per week.
    """

    def __init__(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        typesafe_api_key: Optional[str] = None,
    ):
        self.articles = articles if articles is not None else self._load_articles()
        self._embed_model_name = embed_model_name
        self._embed_model: Optional[SentenceTransformer] = None
        self._typesafe_client: Optional[TypeSafeClient] = None
        self._typesafe_api_key = typesafe_api_key or os.getenv("TYPESAFE_API_KEY")

    def _load_articles(self) -> List[Dict[str, Any]]:
        """Loads articles from articles.json if it exists."""
        if articles_path.exists():
            try:
                with open(articles_path, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                filter_logger.info(f"Loaded {len(articles)} articles from {articles_path}")
                return articles
            except Exception as e:
                filter_logger.error(f"Failed to read articles from {articles_path}: {e}")
                return []
        filter_logger.warning(f"{articles_path} does not exist.")
        return []

    @property
    def embed_model(self) -> SentenceTransformer:
        """Lazy-loaded SentenceTransformer for highlight semantic similarity."""
        if self._embed_model is None:
            filter_logger.info(f"Loading SentenceTransformer model: {self._embed_model_name}")
            self._embed_model = SentenceTransformer(self._embed_model_name)
        return self._embed_model

    @property
    def typesafe_client(self) -> TypeSafeClient:
        """Lazy-loaded TypeSafe AI client."""
        if self._typesafe_client is None:
            if not self._typesafe_api_key:
                filter_logger.warning("TYPESAFE_API_KEY is not set. TypeSafe AI calls may fail.")
            self._typesafe_client = TypeSafeClient(api_key=self._typesafe_api_key)
        return self._typesafe_client

    def drop_short_articles(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        min_char: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Check whether the article has a certain amount of text.
        Articles must have length greater than min_char (default: 500 characters).
        Drops articles with missing title, missing URL, or content <= min_char.
        """
        source_articles = articles if articles is not None else self.articles
        filtered: List[Dict[str, Any]] = []
        dropped_count = 0

        for article in source_articles:
            title = (article.get('title') or '').strip()
            url = (article.get('url') or '').strip()
            content = article.get('content') or ''
            content_str = str(content).strip()

            if not title or not url or len(content_str) <= min_char:
                dropped_count += 1
                filter_logger.debug(
                    f"Dropped short/empty article '{title}' (content length: {len(content_str)} <= {min_char})"
                )
                continue
            filtered.append(article)

        filter_logger.info(
            f"Step 1 (Length filter > {min_char} chars): Kept {len(filtered)}/{len(source_articles)} articles "
            f"({dropped_count} dropped)."
        )
        return filtered

    # Backwards-compatibility alias matching initial draft signature
    def _drop_empty(self, min_char: int = 500) -> List[Dict[str, Any]]:
        return self.drop_short_articles(min_char=min_char)

    def dedup_by_url(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step 2: Removes duplicate articles coming from the same URL.
        Normalizes the URL and keeps only the first occurrence.
        """
        source_articles = articles if articles is not None else self.articles
        unique_articles: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        dropped_count = 0

        for article in source_articles:
            raw_url = (article.get('url') or '').strip()
            if not raw_url:
                continue

            # Normalize URL: strip trailing slash and lowercase scheme + netloc
            try:
                parsed = urlparse(raw_url)
                norm_path = parsed.path.rstrip('/')
                normalized_url = urlunparse((
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    norm_path,
                    parsed.params,
                    parsed.query,
                    ''  # ignore fragment
                ))
            except Exception:
                normalized_url = raw_url.rstrip('/')

            if normalized_url in seen_urls:
                dropped_count += 1
                filter_logger.info(f"Duplicate URL dropped: '{raw_url}' ({article.get('title')})")
                continue

            seen_urls.add(normalized_url)
            unique_articles.append(article)

        filter_logger.info(
            f"Step 2 (URL Deduplication): Kept {len(unique_articles)}/{len(source_articles)} articles "
            f"({dropped_count} duplicates dropped)."
        )
        return unique_articles

    def dedup_by_highlight_similarity(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: float = 0.72,
    ) -> List[Dict[str, Any]]:
        """
        Step 3: Checks whether articles report the same topic/event by comparing
        the semantic embeddings of their highlights across publishers.
        Keeps only the first article for each topic.
        """
        source_articles = articles if articles is not None else self.articles
        if len(source_articles) <= 1:
            return list(source_articles)

        # Extract highlight text for each article
        highlights = [self._extract_highlight_text(a) for a in source_articles]
        embeddings = self.embed_model.encode(highlights, normalize_embeddings=True)

        unique_articles: List[Dict[str, Any]] = []
        unique_embeddings: List[np.ndarray] = []
        dropped_count = 0

        for i, article in enumerate(source_articles):
            title = article.get('title', '')
            emb = embeddings[i]

            if not unique_embeddings:
                unique_articles.append(article)
                unique_embeddings.append(emb)
                continue

            # Cosine similarity against all already kept article highlights
            sims = np.dot(np.array(unique_embeddings), emb)
            max_sim_idx = int(np.argmax(sims))
            max_sim = float(sims[max_sim_idx])

            if max_sim >= similarity_threshold:
                dropped_count += 1
                matched_title = unique_articles[max_sim_idx].get('title', '')
                filter_logger.info(
                    f"Step 3 (Duplicate highlight topic dropped, sim={max_sim:.3f} >= {similarity_threshold}):\n"
                    f"  Dropped: '{title}'\n"
                    f"  Retained: '{matched_title}'"
                )
                continue

            unique_articles.append(article)
            unique_embeddings.append(emb)

        filter_logger.info(
            f"Step 3 (Highlight Similarity Dedup with threshold {similarity_threshold}): "
            f"Kept {len(unique_articles)}/{len(source_articles)} articles ({dropped_count} dropped)."
        )
        return unique_articles

    @staticmethod
    def _extract_highlight_text(article: Dict[str, Any], max_chars: int = 1500) -> str:
        """
        Extracts the highlight of the article.
        Exa provides `highlight` (either a list of excerpt strings or a string).
        Falls back to initial content if highlight is missing.
        """
        highlight = article.get('highlight') or article.get('highlights')
        highlight_text = ""

        if isinstance(highlight, list):
            highlight_text = "\n".join(str(h).strip() for h in highlight if h).strip()
        elif isinstance(highlight, str):
            highlight_text = highlight.strip()

        if not highlight_text:
            content = article.get('content') or ''
            highlight_text = str(content)[:max_chars].strip()
        else:
            highlight_text = highlight_text[:max_chars]

        return highlight_text

    def classify_article_importance(
        self,
        article: Dict[str, Any],
        model: str = "jev-latest",
    ) -> tuple[bool, float, str]:
        """
        Uses TypeSafe AI (Jev model) to classify whether the article is important for an
        AI engineer / data scientist's weekly AI digest based on its highlight.
        
        Returns:
            (is_important: bool, confidence: float, choice: str)
        """
        title = (article.get('title') or '').strip()
        highlight_text = self._extract_highlight_text(article)

        state = f"Title: {title}\n\nHighlight / Key Findings:\n{highlight_text}"

        questions = {
            "digest_relevance": Choice(
                instructions=(
                    "Classify whether this article should be included in a weekly technical AI digest "
                    "for an AI engineer or data scientist. "
                    "We WANT articles covering: foundation/open-weight model releases, architecture breakthroughs, "
                    "technical AI research papers, core developer tooling/frameworks, AI safety/security evaluations, "
                    "and hardware/inference performance benchmarks. "
                    "We DO NOT WANT: company mergers/acquisitions, corporate business partnerships, "
                    "commercial integrations between existing tools (e.g. tool A integrating into tool B), "
                    "customer case studies, enterprise marketing, executive moves, or general sales announcements."
                ),
                criteria={
                    "include": "Technical AI breakthroughs, model releases, research papers, developer frameworks, benchmarks, or AI safety evaluations.",
                    "exclude": "Company partnerships, corporate integrations between tools, business deals, customer stories, or non-technical business fluff."
                }
            )
        }

        try:
            response = self.typesafe_client.system_one(
                model=model,
                state=state,
                questions=questions,
            )
            choice_ans = response.choices.get("digest_relevance")
            if choice_ans:
                is_important = (choice_ans.choice == "include")
                return is_important, choice_ans.confidence, choice_ans.choice
            return True, 0.5, "unknown"
        except Exception as e:
            filter_logger.error(f"TypeSafe AI Jev classification failed for '{title}': {e}")
            # In case of API failure, default to keeping article so pipeline does not drop news ungracefully
            return True, 0.5, "error_fallback"

    def filter_by_importance(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        model: str = "jev-latest",
    ) -> List[Dict[str, Any]]:
        """
        Step 4: Filters articles using TypeSafe AI (Jev) to retain only substantive,
        technically relevant AI news and eliminate corporate business partnerships/integrations.
        """
        source_articles = articles if articles is not None else self.articles
        important_articles: List[Dict[str, Any]] = []
        dropped_count = 0

        for article in source_articles:
            title = article.get('title', '')
            is_important, confidence, choice = self.classify_article_importance(article, model=model)
            article['is_important'] = is_important
            article['importance_confidence'] = confidence
            article['importance_choice'] = choice

            if is_important:
                filter_logger.info(
                    f"Step 4 (TypeSafe Jev: INCLUDE conf={confidence:.2f}): '{title}'"
                )
                important_articles.append(article)
            else:
                dropped_count += 1
                filter_logger.info(
                    f"Step 4 (TypeSafe Jev: EXCLUDE conf={confidence:.2f}): '{title}' (Partnership/Integration/Fluff)"
                )

        filter_logger.info(
            f"Step 4 (Importance Filter via Jev): Kept {len(important_articles)}/{len(source_articles)} articles "
            f"({dropped_count} dropped)."
        )
        return important_articles

    def cap_and_balance(
        self,
        articles: List[Dict[str, Any]],
        target_count: int = 12,
        max_safety_count: int = 2,
        min_per_category: int = 1,
        max_per_category: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Step 5: Caps filtered articles to ~10-12 articles, balancing across categories
        while prioritizing articles with higher Jev importance confidence.

        Rules:
        - Caps 'safety' / policy articles at `max_safety_count` (default: 2) so governance
          and policy announcements do not crowd out actionable engineering content.
        - Guarantees minimum representation (default: 1) for each category present in the pool.
        - Fills all remaining slots up to `target_count` strictly prioritized by
          `importance_confidence` descending across technical categories (model releases,
          research papers, product/tooling).
        """
        if len(articles) <= target_count:
            return articles

        # Group articles by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for a in articles:
            cat = a.get('category')
            by_category.setdefault(cat, []).append(a)

        # Sort within each category by importance confidence descending
        for cat in by_category:
            by_category[cat].sort(
                key=lambda a: a.get('importance_confidence', 1.0),
                reverse=True
            )

        selected: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {cat: 0 for cat in by_category}

        # Step 5a: Ensure baseline representation per category (1 article each)
        for cat, items in by_category.items():
            if items and len(selected) < target_count:
                article = items.pop(0)
                selected.append(article)
                category_counts[cat] += 1
                filter_logger.info(
                    f"[5a] [{cat}] SELECTED '{article.get('title')}' "
                    f"(conf={article.get('importance_confidence', 1.0):.3f})"
                )

        filter_logger.info(
            f"[5a DONE] {len(selected)} articles in selected after baseline pass. "
            f"Counts: {category_counts}"
        )

        # Step 5b: Fill remaining slots prioritized by Jev importance confidence
        round_num = 0
        while len(selected) < target_count:
            round_num += 1
            filter_logger.info(
                f"\n[5b Round {round_num}] selected={len(selected)}/{target_count}  counts={category_counts}"
            )

            candidates: List[tuple[Dict[str, Any], str]] = []
            for cat, items in by_category.items():
                cat_cap = max_safety_count if cat == 'safety' else max_per_category
                if category_counts[cat] < cat_cap and items:
                    candidates.append((items[0], cat))

            if not candidates:
                filter_logger.info(f"[5b Round {round_num}] No candidates left. Stopping early.")
                break

            # Compute and log sort scores for every candidate before sorting
            filter_logger.info(f"[5b Round {round_num}] Candidates ({len(candidates)}):")
            for article, cat in candidates:
                conf = article.get('importance_confidence', 1.0)
                diversity_score = -category_counts[cat]
                filter_logger.info(
                    f"    [{cat}] '{article.get('title')}' "
                    f"→ score=({conf:.3f}, {diversity_score})  "
                    f"[cat_count={category_counts[cat]}]"
                )

            # Prioritize higher confidence; break ties by picking category with fewer selected items
            candidates.sort(
                key=lambda pair: (
                    pair[0].get('importance_confidence', 1.0),
                    -category_counts[pair[1]]
                ),
                reverse=True
            )

            best_candidate, best_cat = candidates[0]
            best_conf = best_candidate.get('importance_confidence', 1.0)
            best_score = (best_conf, -category_counts[best_cat])

            runner_up_info = ""
            if len(candidates) > 1:
                ru_article, ru_cat = candidates[1]
                ru_conf = ru_article.get('importance_confidence', 1.0)
                ru_score = (ru_conf, -category_counts[ru_cat])
                runner_up_info = f"  runner-up=[{ru_cat}] '{ru_article.get('title')}' score={ru_score}"

            filter_logger.info(
                f"[5b Round {round_num}] WINNER: [{best_cat}] '{best_candidate.get('title')}' "
                f"score={best_score}{runner_up_info}"
            )

            by_category[best_cat].pop(0)
            selected.append(best_candidate)
            category_counts[best_cat] += 1

        filter_logger.info(
            f"Step 5 (Capping & Balancing): Selected {len(selected)} articles (target: {target_count}). "
            f"Breakdown by category: {category_counts}"
        )
        return selected

    def filter_articles(
        self,
        articles: Optional[List[Dict[str, Any]]] = None,
        min_char: int = 500,
        similarity_threshold: float = 0.72,
        target_count: int = 12,
        max_safety_count: int = 2,
        min_per_category: int = 1,
        max_per_category: int = 5,
        apply_importance_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Runs the complete filtering pipeline:
        1. Length filter (> 500 characters)
        2. URL deduplication
        3. Highlight semantic similarity deduplication
        4. Importance classification for AI engineers via TypeSafe AI (Jev)
        5. Balance & cap to target_count (~10 to 12 articles) with safety cap and confidence priority
        """
        source_articles = articles if articles is not None else self.articles
        filter_logger.info(f"Starting ArticleFilter pipeline on {len(source_articles)} raw articles.")

        # Step 1: Min length
        s1 = self.drop_short_articles(source_articles, min_char=min_char)

        # Step 2: URL Dedup
        s2 = self.dedup_by_url(s1)

        # Step 3: Highlight Similarity Dedup
        s3 = self.dedup_by_highlight_similarity(s2, similarity_threshold=similarity_threshold)

        # Step 4: Importance classification (Jev TypeSafe AI)
        if apply_importance_filter:
            s4 = self.filter_by_importance(s3)
        else:
            s4 = s3

        # Step 5: Cap to 10-12 articles
        final_articles = self.cap_and_balance(
            s4,
            target_count=target_count,
            max_safety_count=max_safety_count,
            min_per_category=min_per_category,
            max_per_category=max_per_category,
        )

        filter_logger.info(
            f"ArticleFilter pipeline completed. Final count: {len(final_articles)} articles."
        )
        return final_articles

    def save_filtered_articles(
        self,
        articles: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Path:

        filtered_by_cat = {}
        for a in articles:
            cat = a.get('category')
            filtered_by_cat.setdefault(cat, []).append(a)

        """Saves filtered articles to a JSON file."""
        target_path = output_path or filtered_articles_path
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_by_cat, f, indent=4, ensure_ascii=False)
        filter_logger.info(f"Saved {len(articles)} filtered articles to {target_path}")
        return target_path


if __name__ == "__main__":
    article_filter = ArticleFilter()
    filtered = article_filter.filter_articles(target_count=12)
    article_filter.save_filtered_articles(filtered)

    print(f"\nSuccessfully filtered articles down to {len(filtered)} items:")
    for idx, article in enumerate(filtered, 1):
        cat = article.get('category', 'Uncategorized')
        conf = article.get('importance_confidence', 0.0)
        print(f"{idx}. [{cat}] (Conf: {conf:.2f}) {article['title']}")
        print(f"   URL: {article['url']}\n")