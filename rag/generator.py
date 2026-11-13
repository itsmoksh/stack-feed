"""Grounded answer generation node for the agentic RAG workflow."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from logging_setup import setup_logger
from rag.state import RAGState
from rag.prompts import answer_prompt

load_dotenv()

GENERATOR_MODEL = os.getenv('RAG_GENERATOR_MODEL', 'openai/gpt-oss-120b')
ABSTENTION_ANSWER = 'I could not find enough information in the retrieved articles to answer that.'

generator_logger = setup_logger(
    'agentic_rag_generator',
    level=logging.INFO,
)

generator_model = init_chat_model(
    model=GENERATOR_MODEL,
    model_provider='groq',
    temperature=0,
)


def _message_text(message: Any) -> str:
    """Extract printable text from a LangChain AIMessage."""
    content = getattr(message, 'content', message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return ''.join(
            block.get('text', '') if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content).strip()


def _format_evidence(state: RAGState) -> str:
    """Format the final selected chunks, falling back to retrieved chunks."""
    evidence = state.get('combined_evidence') or state.get('retrieved') or []
    if evidence:
        articles: dict[str, list[dict[str, Any]]] = {}
        article_order = []
        for index, chunk in enumerate(evidence, start=1):
            article_key = chunk.get('title') or chunk.get('source') or f'Article {index}'
            if article_key not in articles:
                articles[article_key] = []
                article_order.append(article_key)
            articles[article_key].append(chunk)

        blocks = []
        for article_key in article_order:
            chunks = articles[article_key]
            first_chunk = chunks[0]
            title = first_chunk.get('title') or 'Unknown'
            source = first_chunk.get('source') or 'Unknown'
            chunk_texts = []
            for chunk in chunks:
                section = chunk.get('section') or 'Unknown'
                chunk_texts.append(f'Section: {section}\n{chunk.get("text", "")}')
            blocks.append(
                f'ARTICLE TITLE: {title}\n'
                f'SOURCE: {source}\n'
                + '\n\n'.join(chunk_texts)
            )
        return '\n\n'.join(blocks)

    contexts = state.get('contexts', [])
    return '\n\n'.join(
        f'[Evidence {index}]\n{context}'
        for index, context in enumerate(contexts, start=1)
        if context
    )


def _build_prompt(
    question: str,
    evidence: str,
    question_type: str | None = None,
    category_scope: str | None = None,
    article_titles: list[str] | None = None,
) -> str:
    return answer_prompt(question, evidence, question_type, category_scope, article_titles)


def _log_generation(state: RAGState, answer: str) -> None:
    generator_logger.info(
        'Generation completed question_type=%s evidence_chunks=%d answer_chars=%d question=%s',
        state.get('question_type'),
        len(state.get('combined_evidence') or state.get('retrieved') or state.get('contexts', [])),
        len(answer),
        state.get('question', ''),
    )


def _prepare_generation(state: RAGState) -> tuple[str, str] | dict[str, Any]:
    """Validate the request and return its prompt, or an abstention update."""
    question = (state.get('question') or '').strip()
    if not question:
        raise ValueError('RAGState.question must contain the user query')

    evidence = _format_evidence(state)
    if not evidence.strip():
        generator_logger.info(
            'Generation abstained question_type=%s reason=no_evidence question=%s',
            state.get('question_type'),
            question,
        )
        return {'answer': ABSTENTION_ANSWER, 'status': 'abstained'}
    article_titles = list(dict.fromkeys(
        chunk.get('title')
        for chunk in (state.get('combined_evidence') or state.get('retrieved') or [])
        if chunk.get('title')
    ))
    return question, _build_prompt(
        question,
        evidence,
        question_type=state.get('question_type'),
        category_scope=state.get('category_scope'),
        article_titles=article_titles,
    )



def generator_node(state: RAGState) -> dict[str, Any]:
    """Generate the final answer for synchronous graph callers such as evals."""
    prepared = _prepare_generation(state)
    if isinstance(prepared, dict):
        return prepared
    _, prompt = prepared
    response = generator_model.invoke(prompt)
    answer = _message_text(response)
    if not answer:
        answer = ABSTENTION_ANSWER

    _log_generation(state, answer)
    return {'answer': answer, 'status': 'generated'}


async def generator_stream_node(state: RAGState) -> dict[str, Any]:
    """Stream answer tokens for async Discord calls and retain the final answer."""
    prepared = _prepare_generation(state)
    if isinstance(prepared, dict):
        return prepared
    _, prompt = prepared

    answer_parts = []
    async for message_chunk in generator_model.astream(prompt):
        content = getattr(message_chunk, 'content', '')
        if isinstance(content, str):
            answer_parts.append(content)
        elif isinstance(content, list):
            answer_parts.extend(
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in content
            )

    answer = ''.join(answer_parts).strip() or ABSTENTION_ANSWER
    _log_generation(state, answer)
    return {'answer': answer, 'status': 'generated'}
