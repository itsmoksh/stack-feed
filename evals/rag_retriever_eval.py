"""Score retriever test cases prepared by the evaluation suite."""

import logging

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric

from evals.harness import build_metric_report
from logging_setup import setup_logger

eval_logger = setup_logger('rag_retriever_eval', level=logging.INFO)


def run(test_cases):
    """Evaluate contextual precision and recall on the supplied test cases."""
    if not test_cases:
        raise ValueError('The suite supplied no retriever test cases')

    eval_logger.info('Running retrieval metrics on %d test cases', len(test_cases))
    result = evaluate(
        test_cases,
        [
            ContextualPrecisionMetric(eval_mode='system_one', threshold=0.7, include_reason=True),
            ContextualRecallMetric(eval_mode='system_one', threshold=0.7, include_reason=True),
        ],
    )
    eval_logger.info('Retriever metrics completed for %d questions', len(test_cases))
    return build_metric_report(result)
