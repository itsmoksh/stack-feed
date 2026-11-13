"""Run retriever and generator evaluations from one golden dataset."""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

# Keep DeepEval's temporary test-run files isolated for each suite execution.
os.environ['DEEPEVAL_CACHE_FOLDER'] = tempfile.mkdtemp(prefix='stack-feed-eval-suite-')
os.environ.setdefault('DEEPEVAL_FILE_SYSTEM', 'READ_ONLY')

from langsmith import Client
from deepeval.test_case import LLMTestCase

from evals.harness import print_summary, publish_metric_report_to_langsmith
from evals.rag_generator_eval import run as run_generator_eval
from evals.rag_retriever_eval import run as run_retriever_eval
from logging_setup import setup_logger
from rag.generator import ABSTENTION_ANSWER
from rag.workflow import build_graph


EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / 'golden_dataset_router_eval.json'
RESULTS_DIR = EVALS_DIR / 'results'
suite_logger = setup_logger('rag_eval_suite', level=logging.INFO)


class RootRunIdCapture(BaseCallbackHandler):
    """Capture the root LangGraph trace ID for a golden question."""

    def __init__(self):
        self.run_id = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            self.run_id = run_id


def load_questions(dataset_path=GOLDEN_PATH):
    """Load and validate the complete golden set."""
    with Path(dataset_path).open(encoding='utf-8') as dataset_file:
        dataset = json.load(dataset_file)

    questions = dataset.get('questions', [])
    if not questions:
        raise ValueError(f'No questions found in golden dataset: {dataset_path}')

    ids = [question.get('id') for question in questions]
    if any(not question_id for question_id in ids) or len(ids) != len(set(ids)):
        raise ValueError('Every golden question must have a unique, non-empty id')


    return dataset, questions


def _langsmith_client():
    if not (os.getenv('LANGSMITH_API_KEY') or os.getenv('LANGCHAIN_API_KEY')):
        return None
    try:
        return Client()
    except Exception as exc:
        suite_logger.exception('LangSmith client initialization failed; continuing without feedback: %s', exc)
        return None


def run_suite(dataset_path=GOLDEN_PATH):
    dataset, questions = load_questions(dataset_path)
    retriever_questions = [question for question in questions if question.get('type') != 'unanswerable']
    generator_questions = questions
    suite_logger.info(
        'Loaded full golden set: graph runs=%d retriever cases=%d generator cases=%d',
        len(questions),
        len(retriever_questions),
        len(generator_questions),
    )
    suite_logger.info('Running each golden query once and reusing its graph state across both evaluations')

    graph = build_graph()
    retriever_cases = []
    generator_cases = []
    run_ids_by_question = {}
    for golden in questions:
        question_id = golden['id']
        suite_logger.info(
            'Running graph id=%s type=%s question=%s',
            question_id,
            golden['type'],
            golden['question'],
        )
        run_capture = RootRunIdCapture()
        state = graph.invoke(
            {
                'question': golden['question'],
            },
            config={'callbacks': [run_capture]},
        )
        run_ids_by_question[question_id] = run_capture.run_id
        contexts = state.get('contexts', [])
        metadata = {
            'id': question_id,
            'type': golden['type'],
            'routed_type': state.get('question_type'),
            'category': state.get('category'),
            'category_scope': state.get('category_scope'),
            'retrieval_rounds': state.get('rounds', 0),
            'searched_queries': state.get('searched_queries', []),
            'evidence_gap_history': state.get('evidence_gap_history', []),
            'retrieved_chunk_count': len(contexts),
        }

        if golden['type'] != 'unanswerable':
            retriever_cases.append(
                LLMTestCase(
                    input=golden['question'],
                    expected_output=golden['expected_answer'],
                    retrieval_context=contexts,
                    metadata=metadata,
                )
            )
        generator_cases.append(
            LLMTestCase(
                input=golden['question'],
                expected_output=golden['expected_answer'],
                actual_output=state.get('answer') or ABSTENTION_ANSWER,
                retrieval_context=contexts,
                metadata={**metadata, 'unanswerable_probe': golden['type'] == 'unanswerable'},
            )
        )

        if golden['type'] != 'unanswerable' and state.get('question_type') != golden['type']:
            suite_logger.warning(
                'Router mismatch id=%s golden=%s routed=%s',
                question_id,
                golden['type'],
                state.get('question_type'),
            )
        suite_logger.info(
            'Graph completed id=%s rounds=%d searched_queries=%d contexts=%d answer_chars=%d',
            question_id,
            state.get('rounds', 0),
            len(state.get('searched_queries', [])),
            len(contexts),
            len(state.get('answer') or ''),
        )

    suite_logger.info('Starting retriever metric evaluation')
    retriever_report = run_retriever_eval(retriever_cases)
    if generator_cases:
        suite_logger.info('Starting generator metric evaluation')
        generator_report = run_generator_eval(generator_cases)
    else:
        generator_report = {}
    reports = {**retriever_report, **generator_report}

    langsmith_client = _langsmith_client()
    if langsmith_client:
        retriever_feedback = publish_metric_report_to_langsmith(
            retriever_report,
            run_ids_by_question,
            langsmith_client,
            suite_logger,
        )
        generator_feedback = publish_metric_report_to_langsmith(
            generator_report,
            run_ids_by_question,
            langsmith_client,
            suite_logger,
        )
        suite_logger.info(
            'Published %d per-question metric scores to LangSmith',
            retriever_feedback + generator_feedback,
        )
    else:
        suite_logger.warning('LangSmith credentials not found; metrics were not attached to LangSmith runs')

    report = {
        'metadata': {
            'created_at': datetime.now().astimezone().isoformat(),
            'golden_dataset': str(Path(dataset_path).resolve()),
            'golden_dataset_week': dataset.get('week'),
            'evaluation_scope': 'full_golden_dataset',
            'question_count': len(questions),
            'retriever_question_count': len(retriever_questions),
            'generator_question_count': len(generator_questions),
            'retriever_question_ids': [question['id'] for question in retriever_questions],
            'generator_question_ids': [question['id'] for question in generator_questions],
            'final_ranking': 'reciprocal_rank_fusion_for_focused_and_multi_article',
        },
        'metrics': reports,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = RESULTS_DIR / f'rag_eval_suite_{timestamp}.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    suite_logger.info('Full report saved to %s', report_path)
    print_summary('RAG evaluation suite', reports)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Stack Feed RAG evaluation suite')
    parser.add_argument('--dataset', type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()
    run_suite(dataset_path=args.dataset)
