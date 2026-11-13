"""Score generator test cases prepared by the evaluation suite."""

import logging

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from evals.harness import build_metric_report
from logging_setup import setup_logger

eval_logger = setup_logger('rag_generator_eval', level=logging.INFO)


def run(test_cases):
    """Evaluate the supplied cases with the RAG-triad metrics."""
    if not test_cases:
        raise ValueError('The suite supplied no generator test cases')

    eval_logger.info('Running generator metrics on %d test cases', len(test_cases))
    result = evaluate(
        test_cases,
        [
            ContextualRelevancyMetric(eval_mode='system_one', threshold=0.7, include_reason=True),
            AnswerRelevancyMetric(eval_mode='system_one', threshold=0.7, include_reason=True),
            FaithfulnessMetric(eval_mode='system_one', threshold=0.7, include_reason=True),
        ],
    )
    eval_logger.info('Generator metrics completed for %d questions', len(test_cases))
    return build_metric_report(result)
