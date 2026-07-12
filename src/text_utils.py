import re
from typing import List


MOJIBAKE_CHARS = "�□■▲▶◀◆◇●○¤¼½¾ðþÿ�"

ABBREVIATION_PATTERN = re.compile(
    r"\b(?:figs?|eqs?|refs?|dr|mr|mrs|ms|prof|vs|approx|e\.g|i\.e|et\s+al)\.",
    flags=re.IGNORECASE,
)
ABBREVIATION_DOT = "\uE000"
TERMINAL_PUNCTUATION = re.compile(r"[.!?。！？][\"'”’\)\]\}]*$")
SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "experimental",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "acknowledgements",
    "references",
    "supplementary information",
}


def clean_text(text: str) -> str:
    """Normalize PDF-extracted text and remove common scientific-PDF artifacts."""
    if not text:
        return ""

    text = text.replace("\r", "\n")

    # Remove PDF broken hyphenation.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Remove common PDF extraction artifacts.
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"[{}]+".format(re.escape(MOJIBAKE_CHARS)), " ", text)

    # Normalize common mojibake fragments seen in formula-heavy PDFs.
    text = text.replace("¼", " ")
    text = text.replace("ð", " ")
    text = text.replace("¤", " ")
    text = text.replace("ÿ", " ")
    text = text.replace("þ", " ")

    # Remove isolated replacement boxes.
    text = text.replace("�", " ")

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize strange spacing around punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    return text.strip()


def _protect_abbreviations(text: str) -> str:
    """Hide abbreviation periods while sentence boundaries are detected."""
    return ABBREVIATION_PATTERN.sub(
        lambda match: match.group(0).replace(".", ABBREVIATION_DOT),
        text,
    )


def _restore_abbreviations(text: str) -> str:
    return text.replace(ABBREVIATION_DOT, ".")


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip().strip(":")
    normalized = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+", "", stripped, flags=re.I)
    lower = normalized.lower()
    words = normalized.split()

    if lower in SECTION_HEADINGS:
        return True
    if len(words) <= 10 and normalized.isupper() and any(ch.isalpha() for ch in normalized):
        return True
    if re.match(r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?\s+", stripped, flags=re.I):
        return len(words) <= 12 and not TERMINAL_PUNCTUATION.search(stripped)
    return False


def reflow_wrapped_text(text: str) -> str:
    """Join PDF hard wraps while preserving paragraphs, headings, and bullets."""
    text = clean_text(text)
    if not text:
        return ""

    protected = _protect_abbreviations(text)
    logical_lines: list[str] = []

    for raw_paragraph in protected.split("\n\n"):
        buffer = ""

        for raw_line in raw_paragraph.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            is_bullet = bool(re.match(r"^(?:[-•·]|\d+[.)])\s+", line))
            if _looks_like_heading(_restore_abbreviations(line)):
                if buffer:
                    logical_lines.append(buffer)
                    buffer = ""
                logical_lines.append(line)
                continue

            if is_bullet:
                if buffer:
                    logical_lines.append(buffer)
                buffer = re.sub(r"^(?:[-•·]|\d+[.)])\s+", "", line)
                continue

            if buffer and TERMINAL_PUNCTUATION.search(buffer):
                logical_lines.append(buffer)
                buffer = line
            elif buffer:
                buffer = f"{buffer} {line}"
            else:
                buffer = line

        if buffer:
            logical_lines.append(buffer)

    return "\n".join(_restore_abbreviations(line) for line in logical_lines)


def is_complete_sentence(sentence: str) -> bool:
    """Return whether a candidate ends like a complete English or Chinese sentence."""
    return bool(TERMINAL_PUNCTUATION.search(sentence.strip()))


def is_noisy_sentence(sentence: str) -> bool:
    """Detect formula-heavy, metadata-heavy, or PDF-artifact-heavy sentences."""
    if not sentence:
        return True

    sentence = sentence.strip()
    words = sentence.split()
    lower_sentence = sentence.lower()
    cjk_char_count = len(re.findall(r"[\u3400-\u9fff]", sentence))
    has_cjk_text = cjk_char_count >= 6

    if not is_complete_sentence(sentence):
        return True

    if has_cjk_text and cjk_char_count < 8:
        return True

    if not has_cjk_text and len(words) < 4:
        return True

    if (has_cjk_text and len(sentence) > 450) or (not has_cjk_text and len(words) > 90):
        return True

    metadata_patterns = [
        "arxiv:",
        "dated:",
        "university",
        "institute",
        "department",
        "copyright",
        "all rights reserved",
        "corresponding author",
        "journal of",
        "published by",
        "downloaded from",
        "accepted manuscript",
        "preprint",
    ]

    if any(pattern in lower_sentence for pattern in metadata_patterns):
        return True

    noisy_markers = [
        "fig.",
        "figs.",
        "figure",
        "eq.",
        "equation",
        "appendix",
        "supplementary",
        "table",
    ]

    # One mention of figure can be acceptable, but many formula/figure references are usually bad.
    if sum(marker in lower_sentence for marker in noisy_markers) >= 2:
        return True

    weird_char_count = sum(sentence.count(ch) for ch in MOJIBAKE_CHARS)
    if weird_char_count > 1:
        return True

    math_symbols = sum(
        sentence.count(symbol)
        for symbol in [
            "≈", "σ", "∆", "Δ", "⊥", "∥", "∞", "∂", "∑", "∫",
            "α", "β", "γ", "θ", "λ", "μ", "π", "ω", "χ", "η",
            "=", "+", "/", "\\", "<", ">", "|", "±", "√", "∇",
        ]
    )

    if math_symbols > 6:
        return True

    # Too many numbers usually means equation, table, page header, or reference noise.
    number_like = len(re.findall(r"\b\d+(\.\d+)?\b", sentence))
    if number_like > 8:
        return True

    # Too many single-character tokens usually means broken formula text.
    single_char_tokens = sum(1 for word in words if len(word) == 1)
    if not has_cjk_text and single_char_tokens / max(len(words), 1) > 0.22:
        return True

    alphabetic_chars = sum(ch.isalpha() for ch in sentence)
    if alphabetic_chars / max(len(sentence), 1) < 0.48:
        return True

    # Filter obvious page/footer lines.
    if re.search(r"^\d+\s*$", sentence):
        return True

    if re.search(r"^\d{5,}-\d+", sentence):
        return True

    return False


def split_sentences(text: str) -> List[str]:
    """Split scientific text after repairing common PDF hard line wraps."""
    text = reflow_wrapped_text(text)
    if not text:
        return []

    raw_candidates = []
    for logical_line in text.splitlines():
        logical_line = logical_line.strip()
        if not logical_line or _looks_like_heading(logical_line):
            continue

        protected = _protect_abbreviations(logical_line)
        start = 0
        for match in re.finditer(
            r"[.!?。！？][\"'”’\)\]\}]*?(?=\s+|$|[\u3400-\u9fff])",
            protected,
        ):
            raw_candidates.append(_restore_abbreviations(protected[start : match.end()]))
            start = match.end()

        remainder = _restore_abbreviations(protected[start:]).strip()
        if remainder:
            raw_candidates.append(remainder)

    sentences = []
    for sentence in raw_candidates:
        sentence = sentence.strip()
        sentence = re.sub(r"\s+", " ", sentence)

        if not is_noisy_sentence(sentence):
            sentences.append(sentence)

    return sentences


def split_passages(text: str, max_words: int = 120) -> List[str]:
    """Build retrieval passages without cutting a sentence at a word limit."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    passages: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            passages.append(" ".join(current))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        passages.append(" ".join(current))

    return passages
