from typing_extensions import TypedDict, Any, Literal

class RAGState(TypedDict, total=False):
    question: str
    category: Literal["model release","product/tooling","safety","research"] | None
    question_type: Literal["single_fact","comparison","multi_article","category_scoped"] | None
    category_scope: Literal["broad", "focused"] | None
    remaining_subqueries: list[str]
    active_query: str
    retrieved: list[dict[str, Any]]
    combined_evidence: list[dict[str, Any]]
    chunk_ids: list[str]
    sources: list[str]
    contexts: list[str]
    evidence_stats: dict[str, Any]
    searched_queries: list[str]
    retrieval_rankings: list[list[str]]
    evidence_gap_history: list[dict[str, Any]]
    last_retrieval_added: int
    answer: str
    rounds: int
    status: str
