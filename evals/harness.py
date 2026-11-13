import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean


def build_metric_report(result):
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    by_metric = defaultdict(list)
    for tr in test_results:
        for m in tr.metrics_data:
            by_metric[m.name].append({
                "question_id": tr.metadata.get("id"),
                "question_type": tr.metadata.get("type"),
                "category_scope": tr.metadata.get("category_scope"),
                "retrieval_rounds": tr.metadata.get("retrieval_rounds"),
                "searched_queries": tr.metadata.get("searched_queries"),
                "evidence_gap_history": tr.metadata.get("evidence_gap_history"),
                "retrieved_chunk_count": tr.metadata.get("retrieved_chunk_count"),
                "score": m.score,
                "success": m.success,
                "reason": m.reason,
            })

    report = {}
    for metric_name, entries in by_metric.items():
        # group this metric's entries by question type
        grouped = defaultdict(list)
        for e in entries:
            grouped[e["question_type"]].append(e)

        by_type_summary = {}
        for qtype, group in grouped.items():
            scores = [e["score"] for e in group]
            by_type_summary[qtype] = {
                "count": len(scores),
                "average": round(mean(scores), 4),
                "min": min(scores),
                "max": max(scores),
                "pass_rate": round(sum(e["success"] for e in group) / len(group), 4),
            }

        scoped_entries = [e for e in entries if e.get("category_scope") in {"broad", "focused"}]
        by_scope_summary = {}
        for scope in ("broad", "focused"):
            group = [e for e in scoped_entries if e["category_scope"] == scope]
            if not group:
                continue
            scores = [e["score"] for e in group]
            by_scope_summary[scope] = {
                "count": len(scores),
                "average": round(mean(scores), 4),
                "min": min(scores),
                "max": max(scores),
                "pass_rate": round(sum(e["success"] for e in group) / len(group), 4),
            }

        all_scores = [e["score"] for e in entries]
        report[metric_name] = {
            "overall": {
                "count": len(all_scores),
                "average": round(mean(all_scores), 4),
                "min": min(all_scores),
                "max": max(all_scores),
                "pass_rate": round(sum(e["success"] for e in entries) / len(entries), 4),
            },
            "by_type": by_type_summary,
            "by_category_scope": by_scope_summary,
            "questions": entries,
        }
    return report


def publish_metric_report_to_langsmith(report, run_ids_by_question, client, logger):
    """Publish every per-question metric in a built report to its LangSmith run.

    The report is the common interface for eval scripts and a future eval suite;
    callers do not need to inspect DeepEval's result objects themselves.
    """
    published_count = 0
    for metric_name, metric_report in report.items():
        feedback_key = re.sub(r'[^a-z0-9]+', '_', metric_name.lower()).strip('_')
        for question_metric in metric_report.get('questions', []):
            question_id = question_metric.get('question_id')
            run_id = run_ids_by_question.get(question_id)
            score = question_metric.get('score')
            if not run_id:
                logger.warning(
                    'No LangSmith run id captured for id=%s metric=%s; skipping feedback',
                    question_id,
                    feedback_key,
                )
                continue
            if score is None:
                logger.warning(
                    'No score in metric report for id=%s metric=%s; skipping feedback',
                    question_id,
                    feedback_key,
                )
                continue

            reason = question_metric.get('reason') or ''
            comment = (
                f'Question ID: {question_id}\n'
                f"Question type: {question_metric.get('question_type')}\n"
                f"Retrieval rounds: {question_metric.get('retrieval_rounds')}\n"
                f"Retrieved chunks: {question_metric.get('retrieved_chunk_count')}\n"
                f"Searches: {question_metric.get('searched_queries')}\n"
                f'{reason}'
            ).strip()
            try:
                client.create_feedback(
                    run_id=run_id,
                    key=feedback_key,
                    score=float(score),
                    comment=comment,
                )
                published_count += 1
                logger.info(
                    'Published LangSmith feedback id=%s key=%s score=%.4f',
                    question_id,
                    feedback_key,
                    float(score),
                )
            except Exception:
                logger.exception(
                    'Could not publish LangSmith feedback id=%s key=%s',
                    question_id,
                    feedback_key,
                )
    return published_count

def print_summary(title, report):
    """Readable per-metric recap, printed under DeepEval's own report when a file
    is run directly."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    for metric_name, data in report.items():
        overall = data["overall"]
        print(f"\n--- {metric_name} ---")
        print(
            f"  Overall:   avg={overall['average']:.4f}  "
            f"min={overall['min']:.4f}  max={overall['max']:.4f}  "
            f"pass_rate={overall['pass_rate']:.2%}  (n={overall['count']})"
        )

        print("  By type:")
        for qtype, stats in data["by_type"].items():
            print(
                f"    {qtype:<16} avg={stats['average']:.4f}  "
                f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
                f"pass_rate={stats['pass_rate']:.2%}  (n={stats['count']})"
            )

        if data.get("by_category_scope"):
            print("  By category scope:")
            for scope, stats in data["by_category_scope"].items():
                print(
                    f"    {scope:<16} avg={stats['average']:.4f}  "
                    f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
                    f"pass_rate={stats['pass_rate']:.2%}  (n={stats['count']})"
                )

