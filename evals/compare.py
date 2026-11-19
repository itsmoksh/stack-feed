"""Compare a complete RAG evaluation report with the checked-in baseline."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = EVALS_DIR / 'baseline.json'
DEFAULT_RESULTS_DIR = EVALS_DIR / 'results'
METRIC_REPORT_NAMES = {
    'contextual_precision': 'Contextual Precision',
    'contextual_recall': 'Contextual Recall',
    'contextual_relevancy': 'Contextual Relevancy',
    'answer_relevancy': 'Answer Relevancy',
    'faithfulness': 'Faithfulness',
}


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding='utf-8') as json_file:
            value = json.load(json_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Could not read valid JSON from {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return value


def _resolve_report(report_arg: str) -> Path:
    if report_arg != 'latest':
        path = Path(report_arg).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Evaluation report not found: {path}')
        return path

    reports = [
        path for path in DEFAULT_RESULTS_DIR.glob('rag_eval_suite_*.json')
        if '_incomplete_' not in path.name
    ]
    if not reports:
        raise ValueError(f'No complete-looking evaluation reports found in {DEFAULT_RESULTS_DIR}')
    return max(reports, key=lambda path: path.stat().st_mtime)


def _validate_report_against_baseline(report: dict, baseline: dict) -> list[str]:
    errors = []
    metadata = report.get('metadata') or {}
    baseline_root = baseline.get('baseline') or {}
    categories = baseline_root.get('categories') or {}

    if metadata.get('evaluation_scope') != 'full_golden_dataset':
        errors.append(
            'Report is not marked as a complete full-golden-dataset evaluation '
            f'(evaluation_scope={metadata.get("evaluation_scope")!r}).'
        )

    expected_week = baseline_root.get('golden_dataset_week')
    if expected_week and metadata.get('golden_dataset_week') != expected_week:
        errors.append(
            f'Golden dataset week mismatch: report has {metadata.get("golden_dataset_week")!r}; '
            f'baseline expects {expected_week!r}.'
        )

    expected_hash = baseline_root.get('golden_dataset_sha256')
    dataset_path = EVALS_DIR / 'golden_dataset_router_eval.json'
    if expected_hash:
        try:
            actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        except OSError as exc:
            errors.append(f'Could not read golden dataset for compatibility check: {exc}')
        else:
            if actual_hash != expected_hash:
                errors.append(
                    'Golden dataset content differs from the baseline snapshot. '
                    'Refresh the baseline intentionally before comparing this change.'
                )

    expected_question_count = sum(
        category.get('question_count', 0)
        for category in categories.values()
    )
    if metadata.get('question_count') != expected_question_count:
        errors.append(
            f'Question count mismatch: report has {metadata.get("question_count")!r}; '
            f'baseline expects {expected_question_count}.'
        )

    metrics = report.get('metrics') or {}
    for category, category_data in categories.items():
        expected_count = category_data.get('question_count')
        for metric_name in category_data.get('metrics', {}):
            report_metric_name = METRIC_REPORT_NAMES.get(metric_name)
            if not report_metric_name:
                errors.append(f'Baseline contains unsupported metric key: {metric_name}')
                continue
            metric_report = metrics.get(report_metric_name)
            if not metric_report:
                errors.append(f'Report is missing metric: {report_metric_name}')
                continue
            type_summary = (metric_report.get('by_type') or {}).get(category)
            if not type_summary:
                errors.append(f'Report is missing category {category!r} for {report_metric_name}.')
                continue
            if type_summary.get('count') != expected_count:
                errors.append(
                    f'Count mismatch for {category}/{metric_name}: report has '
                    f'{type_summary.get("count")!r}; baseline expects {expected_count}.'
                )
            score = type_summary.get('average')
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                errors.append(f'Invalid average for {category}/{metric_name}: {score!r}.')

    return errors


def compare_report(report: dict, baseline: dict) -> tuple[list[dict], list[str]]:
    errors = _validate_report_against_baseline(report, baseline)
    if errors:
        return [], errors

    results = []
    categories = baseline['baseline']['categories']
    report_metrics = report['metrics']
    for category, category_data in categories.items():
        for metric_name, baseline_metric in category_data['metrics'].items():
            report_metric_name = METRIC_REPORT_NAMES[metric_name]
            score = report_metrics[report_metric_name]['by_type'][category]['average']
            threshold = baseline_metric['pass_threshold']
            operator = threshold['operator']
            threshold_value = threshold['value']
            direction = baseline_metric['direction']

            if direction == 'higher_is_better' and operator == '>=':
                passed = score >= threshold_value
            elif direction == 'lower_is_better' and operator == '<=':
                passed = score <= threshold_value
            else:
                errors.append(
                    f'Unsupported direction/operator for {category}/{metric_name}: '
                    f'{direction!r} {operator!r}.'
                )
                continue

            results.append({
                'category': category,
                'metric': metric_name,
                'score': score,
                'threshold': threshold_value,
                'operator': operator,
                'direction': direction,
                'passed': passed,
            })
    return results, errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare RAG eval aggregates with baseline.json')
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        '--report',
        default='latest',
        help='Report JSON path, or "latest" to select the newest complete report under evals/results.',
    )
    args = parser.parse_args()

    try:
        baseline_path = args.baseline.expanduser().resolve()
        report_path = _resolve_report(args.report)
        baseline = _load_json(baseline_path)
        report = _load_json(report_path)
        results, errors = compare_report(report, baseline)
    except ValueError as exc:
        print(f'BASELINE COMPARISON ERROR: {exc}', file=sys.stderr)
        return 2

    print(f'Baseline: {baseline_path}')
    print(f'Evaluation report: {report_path}')
    print('\nCategory-level regression checks (2 SD thresholds)')
    print(f'{"CATEGORY":<18} {"METRIC":<26} {"SCORE":>9} {"RULE":>5} {"LIMIT":>9}  RESULT')
    print('-' * 82)
    for result in results:
        print(
            f'{result["category"]:<18} {result["metric"]:<26} '
            f'{result["score"]:>9.4f} {result["operator"]:>5} '
            f'{result["threshold"]:>9.4f}  {"PASS" if result["passed"] else "FAIL"}'
        )

    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 2

    failed = [result for result in results if not result['passed']]
    print(f'\nSummary: {len(results) - len(failed)}/{len(results)} category-metric checks passed.')
    if failed:
        print('PR evaluation FAILED: one or more category-metric thresholds were missed.')
        return 1
    print('PR evaluation PASSED: all category-metric thresholds were met.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
