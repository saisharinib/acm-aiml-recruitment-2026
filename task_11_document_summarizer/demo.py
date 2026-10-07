import os
import matplotlib.pyplot as plt
from summarizer import (split_sentences, score_tfidf, score_frequency, top_indices)

os.makedirs("images", exist_ok=True)
with open("sample_input.txt", encoding="utf-8") as f:
    text = f.read()

sentences = split_sentences(text)
k = max(1, round(len(sentences) * 0.3))
in_words = len(text.split())

methods = {
    "tfidf": ("TF-IDF centroid", score_tfidf),
    "freq": ("Word frequency", score_frequency),
}
summaries, chosen = {}, {}

for key, (label, scorer) in methods.items():
    scores = scorer(sentences)
    idx = top_indices(scores, k)
    chosen[key] = set(idx)
    summaries[key] = " ".join(sentences[i] for i in idx)

    with open(f"sample_output_{key}.txt", "w", encoding="utf-8") as f:
        f.write(summaries[key])

    colors = ["tab:red" if i in chosen[key] else "lightgrey" for i in range(len(sentences))]
    plt.figure(figsize=(9, 4))
    plt.bar(range(1, len(sentences) + 1), scores, color=colors)
    plt.xlabel("Sentence number")
    plt.ylabel("Score")
    plt.title(f"{label}: sentence scores (red = selected)")
    plt.savefig(f"images/scores_{key}.png", dpi=150, bbox_inches="tight")
    plt.close()

overlap = len(chosen["tfidf"] & chosen["freq"])
rows = "\n".join(
    f"| {methods[k][0]} | {len(chosen[k])} of {len(sentences)} | "
    f"{len(summaries[k].split())} of {in_words} | {len(summaries[k].split()) / in_words:.0%} |"
    for k in methods
)

readme = f"""# Task 11: Document Summarizer (Extractive)

## Overview
A command-line tool that reads a text document and returns a short summary made of its most important sentences. It is **extractive**: it selects existing sentences and does not generate new text. Two ranking methods are implemented and compared.

## How it works
1. **Sentence splitting:** the text is cleaned and split into sentences with a regular expression (very short fragments are dropped).
2. **Scoring** (two methods):
   - **TF-IDF centroid:** each sentence becomes a TF-IDF vector (stop words removed). The document's average vector, the centroid, represents its overall topic. A sentence scores higher the more similar it is to the centroid (cosine similarity).
   - **Word frequency:** content words are counted across the whole document and normalised by the most frequent one. A sentence scores the average frequency of its words, which avoids favouring long sentences.
3. **Selection:** the top-scoring sentences are kept (30 percent by default) and put back in their **original order** so the summary reads naturally.

## How to Run
```bash
pip install -r requirements.txt
python summarizer.py sample_input.txt
python summarizer.py sample_input.txt --method freq
python summarizer.py sample_input.txt --sentences 3 --output summary.txt
python demo.py
```
Options: `--method tfidf|freq`, `--sentences N`, `--ratio 0.3`, `--output FILE`.

## Sample Input and Output
Input: [sample_input.txt](sample_input.txt), {len(sentences)} sentences and {in_words} words about solar power.

**TF-IDF summary** ([sample_output_tfidf.txt](sample_output_tfidf.txt)):

> {summaries['tfidf']}

**Word-frequency summary** ([sample_output_freq.txt](sample_output_freq.txt)):

> {summaries['freq']}

## Results
| Method | Sentences kept | Words kept | Compression |
|---|---|---|---|
{rows}

The two methods chose {overlap} of the same {k} sentences, so they largely agree on what matters.

![TF-IDF scores](images/scores_tfidf.png)
![Frequency scores](images/scores_freq.png)

## Limitations and Future Work
- The sentence splitter is a simple regex and can be fooled by abbreviations such as Dr. or e.g.
- Extractive summaries can sound choppy, because sentences are copied without rewriting and may refer to earlier context such as pronouns.
- Both methods ignore meaning and word order.
- Next steps: TextRank (a graph-based method), ROUGE evaluation against human summaries, NLTK tokenisation, and an abstractive transformer model.
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print(readme)