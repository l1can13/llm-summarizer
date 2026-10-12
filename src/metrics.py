import re
from collections import Counter

import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

from . import config as C

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)


def to_lines(text):
    # rougeLsum считает LCS по предложениям, разделённым переводом строки
    if "\n" in text:
        return text
    from .data import split_sentences
    return "\n".join(split_sentences(text, min_words=1))


def flat(text):
    return " ".join(text.split())


def rouge(preds, refs):
    rows = []
    for p, r in zip(preds, refs):
        s = _SCORER.score(to_lines(r), to_lines(p))
        rows.append({"ROUGE-1": s["rouge1"].fmeasure, "ROUGE-2": s["rouge2"].fmeasure,
                     "ROUGE-L": s["rougeLsum"].fmeasure})
    return pd.DataFrame(rows) * 100


def bleu(preds, refs):
    return sacrebleu.corpus_bleu([flat(p) for p in preds], [[flat(r) for r in refs]]).score


def bertscore(preds, refs, batch_size=32):
    from bert_score import score
    _, _, f1 = score([flat(p) for p in preds], [flat(r) for r in refs], lang="en", model_type=C.BERTSCORE_MODEL,
                     rescale_with_baseline=True, batch_size=batch_size, verbose=False)
    return f1.numpy() * 100


def words(text):
    return re.findall(r"\w+", text.lower())


def novel_ngram_share(summary, article, n=2):
    # n-граммы резюме считаются внутри предложений, чтобы не учитывать стыки между выбранными предложениями
    s_ng = []
    for line in to_lines(summary).split("\n"):
        w = words(line)
        s_ng += list(zip(*[w[i:] for i in range(n)]))
    if not s_ng:
        return np.nan
    a = words(article)
    a_ng = set(zip(*[a[i:] for i in range(n)]))
    return 100 * sum(g not in a_ng for g in s_ng) / len(s_ng)


def evaluate(name, preds, refs, articles, with_bertscore=True):
    per_doc = rouge(preds, refs)
    per_doc["words"] = [len(words(p)) for p in preds]
    per_doc["novel_2grams"] = [novel_ngram_share(p, a) for p, a in zip(preds, articles)]
    if with_bertscore:
        per_doc["BERTScore"] = bertscore(preds, refs)
    summary = per_doc.mean().to_dict()
    summary["BLEU"] = bleu(preds, refs)
    return pd.Series(summary, name=name), per_doc


# ---------- быстрый ROUGE для разметки «оракула» ----------

def _tokens(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def _f1(cand, ref):
    overlap = sum((cand & ref).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(cand.values()), overlap / sum(ref.values())
    return 2 * p * r / (p + r)


def greedy_oracle(sentences, reference, k=C.TOP_K):
    # Жадный выбор до k предложений, максимизирующих ROUGE-1 + ROUGE-2 с эталоном
    ref = _tokens(reference)
    ref1, ref2 = _ngrams(ref, 1), _ngrams(ref, 2)
    toks = [_tokens(s) for s in sentences]
    s1, s2 = [_ngrams(t, 1) for t in toks], [_ngrams(t, 2) for t in toks]
    selected, best = [], 0.0
    sel1, sel2 = Counter(), Counter()
    for _ in range(k):
        cand_best, cand_idx = best, None
        for i in range(len(sentences)):
            if i in selected:
                continue
            score = _f1(sel1 + s1[i], ref1) + _f1(sel2 + s2[i], ref2)
            if score > cand_best:
                cand_best, cand_idx = score, i
        if cand_idx is None:
            break
        selected.append(cand_idx)
        sel1, sel2 = sel1 + s1[cand_idx], sel2 + s2[cand_idx]
        best = cand_best
    return sorted(selected)
