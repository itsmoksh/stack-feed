"""LangGraph-based agentic retrieval for Stack Feed."""

__all__ = ['RAGState', 'build_graph']


def __getattr__(name):
    """Load graph exports only when requested, keeping node imports lightweight."""
    if name == 'RAGState':
        from .state import RAGState
        return RAGState
    if name == 'build_graph':
        from .workflow import build_graph
        return build_graph
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
