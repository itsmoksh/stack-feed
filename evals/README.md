# RAG Evaluation Suite

Run both evaluations from the shared golden dataset with:

```powershell
python evals/run_suite.py
```

To write a known report path and compare its aggregate-by-question-type scores
with the six-run baseline:

```powershell
python evals/run_suite.py --output evals/results/pr_report.json
python evals/compare.py --report evals/results/pr_report.json
```

The suite runs every question in `golden_dataset_router_eval.json` through the
full graph once, then builds evaluator cases from the same graph result.
Retriever metrics include all answerable questions; unanswerable questions are
excluded from contextual precision and recall because they have no expected
retrieval evidence. Generator metrics include every question, including
unanswerable ones.

Category-scoped queries are included in retriever evaluation.
Broad category queries scroll the extracted category and retain the first two
chunks per article; focused category queries use category-filtered hybrid
retrieval with bounded evidence-gap follow-up searches.
The eval modules only score test cases and do not invoke the graph themselves.

The suite combines the metric reports, prints a summary, writes a timestamped
JSON report under `evals/results/`, and attaches per-question metric feedback to
the corresponding LangSmith graph runs when LangSmith credentials are present
in `.env` or the environment.

The retriever report contains contextual precision and contextual recall. The
generator report contains contextual relevancy, answer relevancy, and
faithfulness.

The PR workflow runs the suite and comparator automatically. It compares each
category-metric average with the one-sided 2-standard-deviation threshold in
`baseline.json`. On pull requests, the workflow reads the baseline from the PR's
base branch so a PR cannot relax the threshold it is being checked against.
Missing metrics, changed golden-dataset content, mismatched case counts, and
incomplete reports fail the comparison.
