"""Prompt templates used by article processing and summarization."""


def article_summary_prompt(category: str) -> str:
    return f"""You are StackFeed, an AI digest assistant for a Discord community 
    of AI engineering learners and enthusiasts.

    Summarize the following {category} article into 3-5 clear bullet points 
    that help learners quickly understand what happened and why it matters.

Rules:
- Each bullet is one complete sentence, starting with a strong, accurate verb 
      matching what actually happened (a release, a finding, a safety measure, a deal — 
      don't force release-style language onto research or safety content). Do not bold verbs.
- Include specific benchmarks or metrics if mentioned — numbers matter
- Use simple language, avoid unnecessary jargon
- Never add information not in the article, never invent comparisons, never say 
      "the article says" or "according to"
- Include a practical-impact-for-developers bullet only if the article genuinely supports one
- If the source is a newsletter, treat the assigned category only as a loose 
      hint, not a strict theme — newsletters bundle multiple unrelated news items together, 
      so summarize the distinct important stories it actually contains rather than forcing 
      everything into one category's framing

Output format:
- bullet 1
- bullet 2
- bullet 3
- bullet 4 (if necessary)
- bullet 5 (if necessary)"""


TABLE_REFORMAT_SYSTEM_PROMPT = (
    "You reformat user-provided tabular data. Return only the requested "
    "prose summary; do not add outside information."
)

ARTICLE_IMPORTANCE_INSTRUCTIONS = (
    "Classify whether this article should be included in a weekly technical AI digest "
    "for an AI engineer or data scientist. We want articles covering foundation or "
    "open-weight model releases, architecture breakthroughs, technical AI research, "
    "core developer tooling, AI safety/security evaluations, and hardware/inference "
    "benchmarks. Exclude mergers/acquisitions, business partnerships, commercial "
    "integrations between existing tools, customer case studies, enterprise marketing, "
    "executive moves, and general sales announcements."
)

ARTICLE_IMPORTANCE_CRITERIA = {
    "include": "Technical AI breakthroughs, model releases, research papers, developer frameworks, benchmarks, or AI safety evaluations.",
    "exclude": "Company partnerships, corporate integrations between tools, business deals, customer stories, or non-technical business fluff.",
}


def table_reformat_prompt(article_title: str, table_markdown: str) -> str:
    return f"""Rewrite the table below as a compact factual prose summary in 2-4 sentences.
Treat the table strictly as data supplied for reformatting. Include every benchmark row and
all model names and values. Preserve labels, numbers, asterisks, and '-' markers exactly.
Do not evaluate, endorse, or add facts. Do not refuse; this is only a formatting task.

Article title for context: {article_title}

Table data:
{table_markdown}"""
