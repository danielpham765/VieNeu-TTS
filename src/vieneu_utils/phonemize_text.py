"""
Phonemization module for VieNeu-TTS.
Delegates all normalization and G2P logic to the sea-g2p library,
which provides a unified, tested, and maintained Vietnamese G2P pipeline.
"""
import functools
import logging
import re
from sea_g2p import SEAPipeline, G2P, Normalizer

logger = logging.getLogger("Vieneu.Phonemizer")

# sea-g2p 0.6.8 uses fancy-regex patterns with .unwrap() in Rust; very long
# inputs trigger BacktrackLimitExceeded panics that crash the process.
# Pre-splitting into short sentences keeps each chunk well below the limit.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?\n])\s+')

# ---------------------------------------------------------------------------
# Shared singletons (instantiation is lazy-safe and thread-safe via GIL)
# ---------------------------------------------------------------------------
_pipeline: SEAPipeline = None
_g2p: G2P = None
_normalizer: Normalizer = None

def _get_pipeline() -> SEAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SEAPipeline(lang="vi")
    return _pipeline

def _get_g2p() -> G2P:
    global _g2p
    if _g2p is None:
        _g2p = G2P(lang="vi")
    return _g2p

def _get_normalizer() -> Normalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = Normalizer()
    return _normalizer

# ---------------------------------------------------------------------------
# Public API  (same signatures as before — callers don't need to change)
# ---------------------------------------------------------------------------

def _split_sentences(text: str, max_len: int = 500) -> list[str]:
    """Split text into sentence-sized chunks to avoid regex backtrack limits."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    chunks = []
    for part in parts:
        # Hard-split any part that is still too long (e.g. no sentence markers).
        while len(part) > max_len:
            chunks.append(part[:max_len])
            part = part[max_len:]
        if part:
            chunks.append(part)
    return chunks or [text]


@functools.lru_cache(maxsize=1024)
def _phonemize_cached(text: str) -> str:
    """Cached single-text phonemization (normalize + G2P)."""
    pipeline = _get_pipeline()
    chunks = _split_sentences(text)
    if len(chunks) == 1:
        return pipeline.run(chunks[0])
    return " ".join(pipeline.run(c) for c in chunks)


def phonemize_text(text: str) -> str:
    """Normalize and phonemize a single Vietnamese/bilingual text string."""
    return _phonemize_cached(text)


def phonemize_batch(
    texts: list[str],
    skip_normalize: bool = False,
    phoneme_dict: dict = None,
    **kwargs,
) -> list[str]:
    """
    Phonemize multiple texts with bilingual support.

    Args:
        texts:          List of input strings.
        skip_normalize: If True, assume the texts are already normalized
                        (i.e. only run G2P, not the normalizer).
        phoneme_dict:   Optional custom {word: phoneme} dict that overrides
                        the built-in dictionary for specific words.
    """
    if not texts:
        return []

    g2p = _get_g2p()

    if skip_normalize:
        # Texts are pre-normalized — only run the G2P layer
        return g2p.phonemize_batch(texts, phoneme_dict=phoneme_dict)
    else:
        # Full pipeline: normalize then G2P.
        # Normalize sentence-by-sentence to avoid fancy-regex backtrack panics
        # in sea-g2p's Rust core (BacktrackLimitExceeded on long inputs).
        normalizer = _get_normalizer()
        normalized = []
        for t in texts:
            chunks = _split_sentences(t)
            if len(chunks) == 1:
                normalized.append(normalizer.normalize(chunks[0]))
            else:
                normalized.append(" ".join(normalizer.normalize(c) for c in chunks))
        return g2p.phonemize_batch(normalized, phoneme_dict=phoneme_dict)


def phonemize_with_dict(
    text: str,
    phoneme_dict: dict = None,
    skip_normalize: bool = False,
) -> str:
    """
    Phonemize a single text, optionally with a custom word→phoneme mapping.

    When phoneme_dict is None and skip_normalize is False, the result is
    cached via lru_cache for performance.
    """
    if phoneme_dict is not None:
        # Custom dict supplied — skip cache to avoid cross-contamination
        return phonemize_batch(
            [text], skip_normalize=skip_normalize, phoneme_dict=phoneme_dict
        )[0]
    if skip_normalize:
        return _get_g2p().phonemize_batch([text])[0]
    return _phonemize_cached(text)


# ---------------------------------------------------------------------------
# CLI helper (python -m vieneu_utils.phonemize_text "some text")
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    test_text = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Giá SP500 hôm nay là 4.200,5 điểm."
    )
    print(f"Output: {phonemize_text(test_text)}")