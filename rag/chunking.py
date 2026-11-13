"""
Structure-aware chunking for Stack Feed.

Splits article markdown by section headers (##, ###, ####) rather than by
semantic similarity or fixed character windows. Oversized sections are
further split on paragraph boundaries and suffixed "(1/2)", "(2/2)", etc.

Key fix included: a level-1 "#" header that duplicates the article's own
title (the masthead line most sources repeat before their real content
starts) is NOT treated as a section boundary. Without this, every article
produces an extra near-empty chunk that's just the title repeated, which
competes for retrieval slots against chunks that actually carry content.
"""

import re
from dataclasses import dataclass, field


_HEADER_PATTERN = re.compile(r'^(#{1,4})\s+(.*)$', re.MULTILINE)


def _normalize(s: str) -> str:
    """Lowercase and strip non-alphanumerics, for loose title matching."""
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _core_title(title: str) -> str:
    """
    Strips a trailing " - Source Name" or " | Source Name" suffix, since many
    sources repeat the title in a "# Title" header WITHOUT the publication
    suffix that appears in the article's own `title` field (e.g. the stored
    title ends in "| AI Weekly" but the in-body header doesn't).
    """
    return re.split(r'\s[-|]\s', title)[0].strip()


def _is_title_duplicate(header_text: str, article_title: str) -> bool:
    norm_header = _normalize(header_text)
    if not norm_header:
        return False

    candidates = {article_title, _core_title(article_title)}
    for candidate in candidates:
        norm_candidate = _normalize(candidate)
        if not norm_candidate:
            continue
        if norm_header == norm_candidate:
            return True
        # containment either direction catches minor trailing/leading
        # differences (trailing punctuation, a stray space, etc.)
        shorter, longer = sorted([norm_header, norm_candidate], key=len)
        if shorter and shorter in longer and len(shorter) >= 0.8 * len(longer):
            return True
    return False


@dataclass
class Chunk:
    """One chunk produced by StructureAwareChunker."""
    header: str                # e.g. "TL;DR" or "Cybersecurity performance (1/2)"
    text: str                  # original, clean content -- use this for generation/display
    embed_text: str            # text + title/category prefix -- use this for embedding only
    article_title: str = ""
    category: str = ""
    chunk_index: int = 0


class StructureAwareChunker:
    """
    Usage:
        chunker = StructureAwareChunker(max_section_chars=800)
        chunks = chunker.chunk(
            article_text=article["content"],   # should already be table-preprocessed
            article_title=article["title"],
            category=category,
        )
        for c in chunks:
            # c.embed_text  -> pass to the embedding model (dense + sparse)
            # c.text        -> store in payload, use for generation
    """

    def __init__(self, max_section_chars: int = 800, min_chunk_chars: int = 30):
        self.max_section_chars = max_section_chars
        self.min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def chunk(self, article_text: str, article_title: str, category: str = "") -> list[Chunk]:
        sections = self._parse_sections(article_text, article_title)

        chunks: list[Chunk] = []
        idx = 0
        for section in sections:
            for sub in self._split_long_section(section):
                content = sub["content"].strip()
                if len(content) < self.min_chunk_chars:
                    continue  # drop near-empty fragments (e.g. a lone byline)

                embed_text = self._build_embed_text(
                    article_title=article_title,
                    category=category,
                    header=sub["header"],
                    content=content,
                )
                chunks.append(Chunk(
                    header=sub["header"],
                    text=content,
                    embed_text=embed_text,
                    article_title=article_title,
                    category=category,
                    chunk_index=idx,
                ))
                idx += 1

        return chunks

    # ------------------------------------------------------------------ #
    # Section parsing
    # ------------------------------------------------------------------ #

    def _parse_sections(self, markdown_text: str, article_title: str) -> list[dict]:
        matches = list(_HEADER_PATTERN.finditer(markdown_text))

        # Drop level-1 "#" headers that just repeat the article title --
        # these are masthead lines, not real section boundaries.
        matches = [
            m for m in matches
            if not (m.group(1) == '#' and _is_title_duplicate(m.group(2), article_title))
        ]

        if not matches:
            return [{"header": "Introduction", "content": markdown_text.strip()}]

        sections = []

        # Content before the first real header (often byline / title-repeat lines)
        if matches[0].start() > 0:
            intro_blob = markdown_text[:matches[0].start()]
            intro = self._strip_title_duplicate_lines(intro_blob, article_title)
            if intro:
                sections.append({"header": "Introduction", "content": intro})

        for i, m in enumerate(matches):
            header_text = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
            content = markdown_text[start:end].strip()
            if content:
                sections.append({"header": header_text, "content": content})

        return sections

    def _strip_title_duplicate_lines(self, blob: str, article_title: str) -> str:
        """Removes any standalone line in `blob` that's just the article title repeated."""
        lines = [
            line for line in blob.split("\n")
            if line.strip() and not _is_title_duplicate(line.strip(), article_title)
        ]
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ #
    # Oversized-section splitting
    # ------------------------------------------------------------------ #

    def _split_long_section(self, section: dict) -> list[dict]:
        content = section["content"]
        if len(content) <= self.max_section_chars:
            return [section]

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        parts, current = [], ""
        for p in paragraphs:
            if len(current) + len(p) + 2 <= self.max_section_chars:
                current = f"{current}\n\n{p}" if current else p
            else:
                if current:
                    parts.append(current)
                current = p
        if current:
            parts.append(current)

        if not parts:
            # no paragraph breaks at all (one giant block) -- fall back to a
            # hard character split so nothing exceeds max_section_chars by much
            parts = [
                content[i:i + self.max_section_chars]
                for i in range(0, len(content), self.max_section_chars)
            ]

        if len(parts) == 1:
            return [{"header": section["header"], "content": parts[0]}]

        return [
            {"header": f"{section['header']} ({i + 1}/{len(parts)})", "content": part}
            for i, part in enumerate(parts)
        ]

    # ------------------------------------------------------------------ #
    # Embedding text construction
    # ------------------------------------------------------------------ #

    def _build_embed_text(self, article_title: str, category: str, header: str, content: str) -> str:
        prefix = f"[{category}] " if category else ""
        return f"{prefix}{article_title} — {header}: {content}"
