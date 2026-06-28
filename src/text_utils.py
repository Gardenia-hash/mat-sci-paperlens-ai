import re
from typing import List


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter for English scientific text."""
    text = clean_text(text)
    if not text:
        return []

    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    sentences = []
    for sentence in candidates:
        sentence = sentence.strip()
        if len(sentence.split()) >= 6:
            sentences.append(sentence)
    return sentences


def split_passages(text: str, max_words: int = 120) -> List[str]:
    """Split text into paragraph-like passages with a maximum word count."""
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= max_words:
            passages.append(paragraph)
        else:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                if len(chunk.split()) >= 10:
                    passages.append(chunk)

    if not passages:
        passages = split_sentences(text)

    return passages
