"""LangGraph node functions for routing and retrieving RAG evidence."""

import os
import json
import logging
from typing import Any

from dotenv import load_dotenv
from pyasn1.type.univ import Null
from typesafe_sdk import Choice, TypeSafeClient

from logging_setup import setup_logger
from rag.hybrid_search import PREFETCH_LIMIT, hybrid_search, reciprocal_rank_fuse
from rag.category_search import scroll_category_chunks, select_initial_chunks_per_article
from rag.state import RAGState
from langchain.chat_models import init_chat_model
from rag.prompts import (
    CATEGORY_CHOICE_CRITERIA,
    CATEGORY_CHOICE_INSTRUCTIONS,
    CATEGORY_SCOPE_CRITERIA,
    CATEGORY_SCOPE_INSTRUCTIONS,
    ROUTER_CHOICE_CRITERIA,
    ROUTER_CHOICE_INSTRUCTIONS,
    category_router_prompt,
    decomposer_prompt,
    evidence_gap_prompt,
    router_prompt,
)

load_dotenv()

if os.getenv('LANGSMITH_AGENT_ID'):
    os.environ.setdefault('LANGSMITH_AGENT_ENVIRONMENT', 'development')

retrieval_logger = setup_logger(
    'agentic_rag_nodes',
    level=logging.INFO,
)

_typesafe_client: TypeSafeClient | None = None
DECOMPOSER_MODEL = 'openai/gpt-oss-20b'
MAX_SUBQUERIES = 5
SINGLE_FACT_TOP_K = 5
COMPARISON_TOP_K = 5
BROAD_CANDIDATE_TOP_K = 20
CATEGORY_BROAD_CHUNKS_PER_ARTICLE = 2
EVIDENCE_GAP_TOP_K = 5
MAX_EVIDENCE_GAP_ROUNDS = 3

model = init_chat_model(
    model = DECOMPOSER_MODEL,
    model_provider='groq',
    temperature=0
)

def _get_typesafe_client() -> TypeSafeClient:
    """Initialize Jev's client lazily, on the first routing request."""
    global _typesafe_client
    if _typesafe_client is None:
        api_key = os.getenv('TYPESAFE_API_KEY')
        if not api_key:
            raise RuntimeError('TYPESAFE_API_KEY is not set')
        _typesafe_client = TypeSafeClient(api_key=api_key)
    return _typesafe_client


def router_node(state: RAGState) -> dict[str, Any]:
    """Classify the requested answer shape without extracting a category."""
    question = (state.get('question') or '').strip()
    if not question:
        raise ValueError('RAGState.question must contain the user query')

    response = _get_typesafe_client().system_one(
        model='jev-latest',
        state=router_prompt(question),
        questions={
            'question_type': Choice(
                instructions=ROUTER_CHOICE_INSTRUCTIONS,
                criteria=ROUTER_CHOICE_CRITERIA,
            ),
        },
    )

    type_answer = response.choices.get('question_type')
    question_type = type_answer.choice if type_answer else None
    if question_type in ['none',Null]:
        question_type = None

    retrieval_logger.info(
        'Router question_type=%s question=%s',
        question_type,
        question,
    )

    return {'question_type': question_type}


def category_node(state: RAGState) -> dict[str, Any]:
    """Extract the requested category and whether the request is broad or focused."""
    question = (state.get('question') or '').strip()
    if not question:
        raise ValueError('RAGState.question must contain the user query')
    if state.get('question_type') != 'category_scoped':
        retrieval_logger.info(
            'Category extraction skipped type=%s question=%s',
            state.get('question_type'),
            question,
        )
        return {'category': None}

    response = _get_typesafe_client().system_one(
        model='jev-latest',
        state=category_router_prompt(question),
        questions={
            'category': Choice(
                instructions=CATEGORY_CHOICE_INSTRUCTIONS,
                criteria=CATEGORY_CHOICE_CRITERIA,
            ),
            'category_scope': Choice(
                instructions=CATEGORY_SCOPE_INSTRUCTIONS,
                criteria=CATEGORY_SCOPE_CRITERIA,
            ),
        },
    )
    category_answer = response.choices.get('category')
    category = category_answer.choice if category_answer else None
    scope_answer = response.choices.get('category_scope')
    category_scope = scope_answer.choice if scope_answer else None
    if category in ['none', Null]:
        category = None
    if category_scope not in ['broad', 'focused']:
        category_scope = 'focused'

    retrieval_logger.info(
        'Category route category=%s scope=%s question=%s',
        category,
        category_scope,
        question,
    )
    return {'category': category, 'category_scope': category_scope}


def category_scroll_node(state: RAGState) -> dict[str, Any]:
    """Scroll category chunks and keep opening chunks for broad category queries."""
    question = (state.get('question') or '').strip()
    if not question:
        raise ValueError('RAGState.question must contain the user query')

    category = state.get('category')
    if category:
        all_category_chunks = scroll_category_chunks(category)
        chunks = select_initial_chunks_per_article(
            all_category_chunks,
            CATEGORY_BROAD_CHUNKS_PER_ARTICLE,
        )
        retrieval_logger.info(
            'Category retrieval completed category=%s scope=%s candidates=%d selected=%d articles=%d',
            category,
            state.get('category_scope'),
            len(all_category_chunks),
            len(chunks),
            len({chunk.get('_article_number') for chunk in chunks}),
        )
    else:
        retrieval_logger.warning(
            'Category was not extracted; falling back to unfiltered hybrid search question=%s',
            question,
        )
        chunks = hybrid_search(question, BROAD_CANDIDATE_TOP_K)

    return {
        'active_query': question,
        'retrieved': chunks,
        'chunk_ids': [chunk['chunk_id'] for chunk in chunks if chunk.get('chunk_id')],
        'sources': list(dict.fromkeys(chunk['source'] for chunk in chunks if chunk.get('source'))),
        'contexts': [chunk['text'] for chunk in chunks if chunk.get('text')],
        'rounds': state.get('rounds', 0) + 1,
        'searched_queries': [question],
        'last_retrieval_added': len(chunks),
        'status': 'retrieved',
    }


def decomposer_node(state: RAGState) -> dict[str, Any]:
    """Split comparisons into focused search queries."""
    question = (state.get('question') or '').strip()
    if not question:
        raise ValueError('RAGState.question must contain the user query')

    retrieval_logger.info(
        'Decomposer started type=%s model=%s question=%s',
        state.get('question_type'),
        DECOMPOSER_MODEL,
        question,
    )
    prompt = decomposer_prompt(question, MAX_SUBQUERIES)

    try:
        response = model.invoke(prompt)
        raw = response.content
        if isinstance(raw, list):
            raw = ''.join(
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in raw
            )
        if not isinstance(raw, str):
            raise TypeError(f'Expected decomposer response text, got {type(raw).__name__}')
        raw = raw.strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw[3:-3].strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        queries = parsed.get('queries', [])
        if not isinstance(queries, list):
            raise ValueError('Decomposer response must contain a list named queries')
        subqueries = list(dict.fromkeys(
            query.strip() for query in queries
            if isinstance(query, str) and query.strip()
        ))[:MAX_SUBQUERIES]
    except Exception as exc:
        retrieval_logger.warning('Decomposition failed; falling back to original query: %s', exc)
        subqueries = []

    if not subqueries:
        subqueries = [question]

    retrieval_logger.info('Decomposer produced %d queries: %s', len(subqueries), subqueries)

    return {
        'remaining_subqueries': subqueries,
        'retrieved': [],
        'chunk_ids': [],
        'sources': [],
        'contexts': [],
        'rounds': 0,
        'searched_queries': [],
        'retrieval_rankings': [],
        'evidence_gap_history': [],
    }


def retriever_node(state: RAGState) -> dict[str, Any]:
    """Retrieve relevant chunks for the active query and accumulate evidence."""
    remaining_subqueries = state.get('remaining_subqueries', [])
    if remaining_subqueries:
        query = remaining_subqueries[0]
        remaining_after_search = remaining_subqueries[1:]
    else:
        query = (state.get('active_query') or state.get('question') or '').strip()
        remaining_after_search = []
    if not query:
        raise ValueError('RAGState must contain an active_query or question')

    question_type = state.get('question_type')
    if question_type == 'multi_article' or (
        question_type == 'category_scoped' and state.get('category_scope') == 'focused'
    ):
        top_k = EVIDENCE_GAP_TOP_K
    elif question_type == 'comparison':
        top_k = COMPARISON_TOP_K
    else:
        top_k = SINGLE_FACT_TOP_K
    category_filter = (
        state.get('category')
        if question_type == 'category_scoped' and state.get('category_scope') == 'focused'
        else None
    )

    retrieval_logger.info(
        'Retrieval started type=%s query=%s category_filter=%s prefetch_limit=%d final_top_k=%d',
        question_type,
        query,
        category_filter,
        PREFETCH_LIMIT,
        top_k,
    )
    new_chunks = hybrid_search(query, top_k, category_filter)
    retrieval_rankings = list(state.get('retrieval_rankings', []))
    if new_chunks:
        retrieval_rankings.append([
            str(chunk.get('chunk_id') or '|'.join(str(chunk.get(field, '')) for field in ('source', 'title', 'text')))
            for chunk in new_chunks
        ])
    accumulated = list(state.get('retrieved', []))
    known_ids = {chunk.get('chunk_id') for chunk in accumulated}
    unique_new_chunks = [
        chunk for chunk in new_chunks
        if chunk.get('chunk_id') not in known_ids
    ]
    accumulated.extend(unique_new_chunks)
    searched_queries = list(state.get('searched_queries', []))
    if query and query not in searched_queries:
        searched_queries.append(query)
    retrieval_logger.info(
        'Retrieval completed query=%s returned=%d accumulated_unique=%d sources=%d',
        query,
        len(new_chunks),
        len(accumulated),
        len({chunk.get('source') for chunk in accumulated if chunk.get('source')}),
    )

    return {
        'active_query': query,
        'remaining_subqueries': remaining_after_search,
        'retrieved': accumulated,
        'chunk_ids': [chunk['chunk_id'] for chunk in accumulated if chunk['chunk_id']],
        'sources': list(dict.fromkeys(chunk['source'] for chunk in accumulated if chunk['source'])),
        'contexts': [chunk['text'] for chunk in accumulated if chunk['text']],
        'rounds': state.get('rounds', 0) + 1,
        'searched_queries': searched_queries,
        'retrieval_rankings': retrieval_rankings,
        'last_retrieval_added': len(unique_new_chunks),
        'status': 'retrieved',
    }


def evidence_gap_node(state: RAGState) -> dict[str, Any]:
    """Decide whether accumulated evidence covers the original request."""
    question = (state.get('question') or '').strip()
    if not question:
        raise ValueError('RAGState.question must contain the user query')

    rounds = state.get('rounds', 0)
    chunks = state.get('retrieved', [])
    history = list(state.get('evidence_gap_history', []))
    searched_queries = list(state.get('searched_queries', []))
    added_last_round = state.get('last_retrieval_added', len(chunks))

    if added_last_round == 0:
        retrieval_logger.info(
            'Evidence gap review stopped reason=no_new_chunks rounds=%d total_chunks=%d',
            rounds,
            len(chunks),
        )
        history.append({
            'round': rounds,
            'stop_reason': 'no_new_chunks',
            'follow_up_query': None,
            'new_chunks_added': 0,
        })
        return {
            'remaining_subqueries': [],
            'evidence_gap_history': history,
            'status': 'evidence_gap_stopped',
        }

    if rounds >= MAX_EVIDENCE_GAP_ROUNDS:
        retrieval_logger.info(
            'Evidence gap review stopped reason=max_rounds rounds=%d total_chunks=%d',
            rounds,
            len(chunks),
        )
        history.append({
            'round': rounds,
            'stop_reason': 'max_rounds',
            'follow_up_query': None,
            'new_chunks_added': added_last_round,
        })
        return {
            'remaining_subqueries': [],
            'evidence_gap_history': history,
            'status': 'evidence_gap_stopped',
        }

    context_inventory = [
        {
            'chunk_id': chunk.get('chunk_id'),
            'title': chunk.get('title'),
            'section': chunk.get('section'),
            'text': chunk.get('text'),
        }
        for chunk in chunks
    ]
    category = state.get('category')
    scope = state.get('category_scope')
    prompt = evidence_gap_prompt(
        question,
        state.get('question_type'),
        scope,
        category,
        searched_queries,
        context_inventory,
    )

    try:
        response = model.invoke(prompt)
        raw = response.content
        if isinstance(raw, list):
            raw = ''.join(
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in raw
            )
        if not isinstance(raw, str):
            raise TypeError(f'Expected evidence-gap response text, got {type(raw).__name__}')
        raw = raw.strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw[3:-3].strip()
            if raw.startswith('json'):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError('Evidence-gap response must be a JSON object')
        enough_evidence = parsed.get('enough_evidence') is True
        missing_points = parsed.get('missing_points', [])
        if not isinstance(missing_points, list):
            missing_points = []
        missing_points = [point.strip() for point in missing_points if isinstance(point, str) and point.strip()]
        follow_up_query = parsed.get('follow_up_query')
        if isinstance(follow_up_query, str):
            follow_up_query = follow_up_query.strip()
        else:
            follow_up_query = None
    except Exception as exc:
        retrieval_logger.warning('Evidence-gap review failed; stopping retrieval: %s', exc)
        enough_evidence = True
        missing_points = []
        follow_up_query = None

    normalized_queries = {query.casefold().strip() for query in searched_queries}
    if follow_up_query and follow_up_query.casefold() in normalized_queries:
        retrieval_logger.info('Evidence-gap follow-up stopped reason=repeated_query query=%s', follow_up_query)
        follow_up_query = None

    if enough_evidence or not follow_up_query:
        stop_reason = 'evidence_sufficient' if enough_evidence else 'no_follow_up_query'
        retrieval_logger.info(
            'Evidence gap review stopped reason=%s rounds=%d chunks=%d missing_points=%s',
            stop_reason,
            rounds,
            len(chunks),
            missing_points,
        )
        history.append({
            'round': rounds,
            'stop_reason': stop_reason,
            'missing_points': missing_points,
            'follow_up_query': None,
            'new_chunks_added': added_last_round,
        })
        return {
            'remaining_subqueries': [],
            'missing_points': missing_points,
            'evidence_gap_history': history,
            'status': 'evidence_gap_stopped',
        }

    retrieval_logger.info(
        'Evidence gap follow-up scheduled round=%d max_rounds=%d missing_points=%s query=%s',
        rounds,
        MAX_EVIDENCE_GAP_ROUNDS,
        missing_points,
        follow_up_query,
    )
    history.append({
        'round': rounds,
        'stop_reason': None,
        'missing_points': missing_points,
        'follow_up_query': follow_up_query,
        'new_chunks_added': added_last_round,
    })
    return {
        'remaining_subqueries': [follow_up_query],
        'missing_points': missing_points,
        'evidence_gap_history': history,
        'status': 'follow_up_scheduled',
    }


def combine_evidence_node(state: RAGState) -> dict[str, Any]:
    """Deduplicate evidence, then fuse multi-query results with reciprocal ranks."""
    candidates = []
    seen = set()
    for chunk in state.get('retrieved', []):
        key = chunk.get('chunk_id') or (
            chunk.get('source', ''),
            chunk.get('title', ''),
            chunk.get('text', ''),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(chunk)

    selection_stats = None
    if state.get('question_type') == 'multi_article' or (
        state.get('question_type') == 'category_scoped'
        and state.get('category_scope') == 'focused'
    ):
        combined, selection_stats = reciprocal_rank_fuse(
            candidates,
            state.get('retrieval_rankings', []),
        )
        retrieval_logger.info(
            'RRF completed type=%s rankings=%d candidates=%d selected=%d',
            state.get('question_type'),
            selection_stats['ranking_count'],
            selection_stats['candidate_count'],
            selection_stats['selected_count'],
        )
    elif state.get('question_type') == 'category_scoped':
        combined = candidates
        selection_stats = {
            'candidate_count': len(candidates),
            'selected_count': len(combined),
            'selected_source_count': len({chunk.get('source') for chunk in combined if chunk.get('source')}),
            'reranker': 'not_applied',
            'category_scope': state.get('category_scope'),
            'selection': 'initial_chunks_per_article',
        }
    else:
        combined = candidates

    if selection_stats is None:
        selection_stats = {
            'candidate_count': len(candidates),
            'selected_count': len(combined),
            'selected_source_count': len({chunk.get('source') for chunk in combined if chunk.get('source')}),
            'reranker': 'not_applied',
        }

    retrieval_logger.info(
        'Evidence selected type=%s candidates=%d selected=%d sources=%d stats=%s',
        state.get('question_type'),
        len(candidates),
        len(combined),
        len({chunk.get('source') for chunk in combined if chunk.get('source')}),
        selection_stats,
    )

    return {
        'combined_evidence': combined,
        'evidence_stats': selection_stats,
        'chunk_ids': [chunk['chunk_id'] for chunk in combined if chunk.get('chunk_id')],
        'sources': list(dict.fromkeys(chunk['source'] for chunk in combined if chunk.get('source'))),
        'contexts': [chunk['text'] for chunk in combined if chunk.get('text')],
        'status': 'evidence_combined',
    }
