"""Extractive document summarizer: TF-IDF centroid ranking and word-frequency ranking."""
import argparse
import re
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text):
    """Split text into sentences using a simple regex."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.split()) >= 4]


def score_tfidf(sentences):
    """Score = cosine similarity between a sentence and the document's average TF-IDF vector."""
    try:
        X = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:  # no usable words
        return np.zeros(len(sentences))
    centroid = np.asarray(X.mean(axis=0))
    return cosine_similarity(X, centroid).ravel()


def score_frequency(sentences):
    """Score = average normalised frequency of a sentence's content words."""
    words_per_sentence = [
        [w for w in re.findall(r"[a-z']+", s.lower())
         if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        for s in sentences
    ]
    freq = Counter(w for words in words_per_sentence for w in words)
    if not freq:
        return np.zeros(len(sentences))
    top = max(freq.values())
    norm = {w: c / top for w, c in freq.items()}
    return np.array([
        sum(norm[w] for w in words) / len(words) if words else 0.0
        for words in words_per_sentence
    ])


def top_indices(scores, n):
    """Indices of the n highest-scoring sentences, returned in original order."""
    best = np.argsort(-np.asarray(scores), kind="stable")[:n]
    return sorted(int(i) for i in best)


def summarize(text, method="tfidf", num_sentences=None, ratio=0.3):
    """Return an extractive summary of `text`."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if num_sentences is None:
        num_sentences = max(1, round(len(sentences) * ratio))
    num_sentences = min(num_sentences, len(sentences))
    scores = score_tfidf(sentences) if method == "tfidf" else score_frequency(sentences)
    return " ".join(sentences[i] for i in top_indices(scores, num_sentences))


def main():
    parser = argparse.ArgumentParser(description="Extractive document summarizer")
    parser.add_argument("file", help="path to a .txt file")
    parser.add_argument("--method", choices=["tfidf", "freq"], default="tfidf")
    parser.add_argument("--sentences", type=int, help="number of sentences to keep")
    parser.add_argument("--ratio", type=float, default=0.3,
                        help="fraction of sentences to keep (default 0.3)")
    parser.add_argument("--output", help="optional file to save the summary")
    args = parser.parse_args()

    try:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        sys.exit(f"File not found: {args.file}")

    summary = summarize(text, args.method, args.sentences, args.ratio)
    print(summary)
    print(f"\n[{len(text.split())} words -> {len(summary.split())} words]")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)


if __name__ == "__main__":
    main()