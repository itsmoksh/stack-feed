"""Qdrant hybrid search utilities for the agentic RAG workflow."""

from qdrant_client.models import models

PREFETCH_LIMIT = 20
RRF_K = 60


def _chunk_rank_key(chunk: dict) -> str:
    """Return a stable identity for matching a chunk across retrieval rounds."""
    chunk_id = chunk.get('chunk_id')
    if chunk_id:
        return str(chunk_id)
    return '|'.join(str(chunk.get(field, '')) for field in ('source', 'title', 'text'))


def reciprocal_rank_fuse(
    chunks: list[dict],
    ranked_chunk_ids: list[list[str]],
    rrf_k: int = RRF_K,
) -> tuple[list[dict], dict]:
    """Fuse per-query rankings with RRF, retaining every unique retrieved chunk."""
    scores: dict[str, float] = {}
    for ranking in ranked_chunk_ids:
        seen_in_ranking = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    fused = []
    seen_chunks = set()
    for chunk in chunks:
        key = _chunk_rank_key(chunk)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        fused.append({**chunk, 'rrf_score': scores.get(key, 0.0)})
    fused.sort(key=lambda chunk: chunk['rrf_score'], reverse=True)
    return fused, {
        'reranker': 'reciprocal_rank_fusion',
        'rrf_k': rrf_k,
        'ranking_count': len(ranked_chunk_ids),
        'candidate_count': len(chunks),
        'selected_count': len(fused),
        'selected_source_count': len({chunk.get('source') for chunk in fused if chunk.get('source')}),
    }


def hybrid_search(query: str, top_k: int, category: str | None = None) -> list[dict]:
    """Search dense and BM25 vectors with RRF and an optional category filter."""
    from rag.vector_store import (
        client,
        collection_name,
        dense_model_name,
        sparse_model_name,
    )

    category_filter = None
    if category:
        category_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key='category',
                    match=models.MatchValue(value=category),
                )
            ]
        )

    points = client.query_points(
        collection_name=collection_name,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=dense_model_name),
                using='dense',
                filter=category_filter,
                limit=PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=sparse_model_name),
                using='sparse',
                filter=category_filter,
                limit=PREFETCH_LIMIT,
            ),
        ],
        query_filter=category_filter,
        limit=top_k,
    ).points

    return [
        {
            'chunk_id': point.payload.get('chunk_id', ''),
            'title': point.payload.get('title', ''),
            'section': point.payload.get('section', ''),
            'category': point.payload.get('category', ''),
            'source': point.payload.get('source', ''),
            'text': point.payload.get('text', ''),
            'retrieval_score': float(point.score) if point.score is not None else 0.0,
        }
        for point in points
    ]
