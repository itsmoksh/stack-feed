"""LangGraph assembly for the agentic RAG nodes currently implemented."""

from langgraph.graph import END, START, StateGraph

from rag.generator import generator_node, generator_stream_node
from rag.retriever import (
    category_node,
    category_scroll_node,
    combine_evidence_node,
    decomposer_node,
    evidence_gap_node,
    retriever_node,
    router_node,
)
from rag.state import RAGState


def route_by_question_type(state: RAGState) -> str:
    """Choose a follow-up step after classifying the requested answer shape."""
    routes = {
        'single_fact': 'retriever',
        'category_scoped': 'category',
        'comparison': 'decomposer',
        'multi_article': 'retriever',
    }
    question_type = state.get('question_type')
    # An unclear or unanswerable query should still attempt ordinary retrieval;
    # the generator can abstain if the returned evidence does not support it.
    return routes.get(question_type, 'retriever')


def route_after_retrieval(state: RAGState) -> str:
    """Route evidence-gap queries through bounded iterative retrieval."""
    question_type = state.get('question_type')
    if question_type == 'comparison':
        return 'retrieve_more' if state.get('remaining_subqueries') else 'combine'
    if question_type == 'multi_article':
        return 'evidence_gap'
    if question_type == 'category_scoped' and state.get('category_scope') == 'focused':
        return 'evidence_gap'
    if state.get('remaining_subqueries'):
        return 'retrieve_more'
    if state.get('question_type') in ['single_fact', None]:
        return 'generate'
    return 'combine'


def route_after_category(state: RAGState) -> str:
    """Use category scroll for broad scope and hybrid retrieval for focused scope."""
    return 'category_scroll' if state.get('category_scope') == 'broad' else 'retriever'


def route_after_evidence_gap(state: RAGState) -> str:
    """Run the targeted query when one is available; otherwise combine evidence."""
    return 'retrieve_more' if state.get('remaining_subqueries') else 'combine'


def build_graph(include_generator: bool = True, stream_generator: bool = False):
    """Build the workflow, optionally using token streaming for answer generation."""
    graph = StateGraph(RAGState)
    graph.add_node('router', router_node)
    graph.add_node('category', category_node)
    graph.add_node('category_scroll', category_scroll_node)
    graph.add_node('retriever', retriever_node)
    graph.add_node('decomposer', decomposer_node)
    graph.add_node('evidence_gap', evidence_gap_node)
    graph.add_node('combine_evidence', combine_evidence_node)
    if include_generator:
        graph.add_node(
            'generator',
            generator_stream_node if stream_generator else generator_node,
        )
    graph.add_edge(START, 'router')
    graph.add_conditional_edges(
        'router',
        route_by_question_type,
        {
            'retriever': 'retriever',
            'category': 'category',
            'decomposer': 'decomposer',
        },
    )
    graph.add_conditional_edges(
        'category',
        route_after_category,
        {
            'category_scroll': 'category_scroll',
            'retriever': 'retriever',
        },
    )
    graph.add_edge('category_scroll', 'combine_evidence')
    graph.add_edge('decomposer', 'retriever')
    graph.add_conditional_edges(
        'retriever',
        route_after_retrieval,
        {
            'retrieve_more': 'retriever',
            'combine': 'combine_evidence',
            'evidence_gap': 'evidence_gap',
            'generate': 'generator' if include_generator else END,
        },
    )
    graph.add_conditional_edges(
        'evidence_gap',
        route_after_evidence_gap,
        {
            'retrieve_more': 'retriever',
            'combine': 'combine_evidence',
        },
    )
    graph.add_edge('combine_evidence', 'generator' if include_generator else END)
    if include_generator:
        graph.add_edge('generator', END)
    return graph.compile()
