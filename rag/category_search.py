"""Category-wide Qdrant scrolling and chunk ordering helpers."""

from collections import defaultdict
import logging
from typing import Any

from qdrant_client.models import models

from logging_setup import setup_logger


CATEGORY_SCROLL_PAGE_SIZE = 256
category_logger = setup_logger('category_scroll', level=logging.INFO)


def parse_article_and_chunk_order(chunk_id: str) -> tuple[int, int] | None:
    """Read article and chunk positions from IDs like `research_article_2_chunk_4`."""
    try:
        article_prefix, chunk_number = chunk_id.rsplit('_chunk_', 1)
        _, article_number = article_prefix.rsplit('_article_', 1)
        return int(article_number), int(chunk_number)
    except (AttributeError, ValueError):
        return None


def group_chunks_by_article(chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group category results by encoded article number and sort chunks in order."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for position, chunk in enumerate(chunks):
        parsed_order = parse_article_and_chunk_order(chunk.get('chunk_id', ''))
        if parsed_order is not None:
            article_number, chunk_number = parsed_order
            chunk['_article_number'] = article_number
            chunk['_chunk_number'] = chunk_number
            group_key = ('article_number', str(article_number))
        else:
            chunk['_article_number'] = position
            chunk['_chunk_number'] = position
            group_key = ('source', chunk.get('source') or chunk.get('title') or str(position))
        grouped[group_key].append(chunk)

    article_groups = list(grouped.values())
    for article_chunks in article_groups:
        article_chunks.sort(key=lambda chunk: chunk['_chunk_number'])
    article_groups.sort(key=lambda article_chunks: article_chunks[0]['_article_number'])
    return article_groups


def scroll_category_chunks(category: str) -> list[dict[str, Any]]:
    """Scroll every stored chunk for one category, following all Qdrant pages."""
    if not category:
        raise ValueError('A category is required for category-scoped scrolling')

    from rag.vector_store import client, collection_name

    category_filter = models.Filter(
        must=[
            models.FieldCondition(
                key='category',
                match=models.MatchValue(value=category),
            )
        ]
    )

    chunks = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=category_filter,
            limit=CATEGORY_SCROLL_PAGE_SIZE,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get('chunk_id', '')
            order = parse_article_and_chunk_order(chunk_id)
            chunks.append(
                {
                    'chunk_id': chunk_id,
                    'title': payload.get('title', ''),
                    'section': payload.get('section', ''),
                    'category': payload.get('category', category),
                    'source': payload.get('source', ''),
                    'text': payload.get('text', ''),
                    'retrieval_score': 0.0,
                    '_article_number': order[0] if order else None,
                    '_chunk_number': order[1] if order else None,
                }
            )

        if next_offset is None:
            break
        offset = next_offset

    article_groups = group_chunks_by_article(chunks)
    ordered_chunks = [chunk for group in article_groups for chunk in group]
    category_logger.info(
        'Category scroll completed category=%s chunks=%d articles=%d pagesize=%d',
        category,
        len(ordered_chunks),
        len(article_groups),
        CATEGORY_SCROLL_PAGE_SIZE,
    )
    return ordered_chunks


def select_initial_chunks_per_article(
    chunks: list[dict[str, Any]],
    chunks_per_article: int = 2,
) -> list[dict[str, Any]]:
    """Keep the first ordered chunks from every article, preserving article order."""
    article_groups = group_chunks_by_article(chunks)
    return [
        chunk
        for article_chunks in article_groups
        for chunk in article_chunks[:chunks_per_article]
    ]
