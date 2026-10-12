import hashlib
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C


def prepare_dataset():
    if not RAW_FILES_OK():
        from huggingface_hub import snapshot_download
        snapshot_download(C.HF_DATASET, repo_type="dataset", allow_patterns=["3.0.0/*", "README.md"],
                          local_dir=str(C.DATA_ROOT))


def RAW_FILES_OK():
    return all(list(C.RAW_DIR.glob(f"{s}-*.parquet")) for s in C.SPLITS)


def load_split(split):
    files = sorted(C.RAW_DIR.glob(f"{split}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)[["id", "article", "highlights"]]


# ---------- очистка ----------

_DM_HEADER = [
    re.compile(r"^By \..{0,250}?UPDATED: \. .{0,40}?\d{4} \. ", re.S),
    re.compile(r"^By \..{0,250}?PUBLISHED: \. .{0,40}?\d{4} \. ", re.S),
    re.compile(r"^By \..{0,200}?Last updated at .{0,60}?\d{4} \. ", re.S),
    re.compile(r"^By \. [^.]{1,80} \. (?:and \. [^.]{1,80} \. )*"),
]
_CNN = re.compile(r"^(?:[A-Z][A-Z .'-]{1,40}(?:, [A-Za-z .'-]{1,30})?\s)?\(CNN\)\s*(?:--\s*)?")


def clean_article(text):
    text = text.strip()
    for p in _DM_HEADER:
        new = p.sub("", text, count=1)
        if new != text:
            text = new
            break
    text = _CNN.sub("", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def has_header(text):
    return text.startswith("By .") or "PUBLISHED: ." in text[:400] or bool(_CNN.match(text))


def boilerplate_share(articles):
    a = pd.Series(articles)
    return {
        "шапка Daily Mail («By . …»), %": 100 * a.str.startswith("By .").mean(),
        "метка «(CNN)» в начале, %": 100 * a.map(lambda t: bool(_CNN.match(t))).mean(),
    }


# ---------- предложения ----------

_PUNKT_READY = False


def _ensure_punkt():
    # Данные токенизатора предложений NLTK хранятся в окружении проекта; при отсутствии скачиваются
    global _PUNKT_READY
    if _PUNKT_READY:
        return
    import sys
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        target = Path(sys.prefix) / "nltk_data"
        nltk.download("punkt_tab", download_dir=str(target), quiet=True)
        if str(target) not in nltk.data.path:
            nltk.data.path.append(str(target))
    _PUNKT_READY = True


def split_sentences(text, min_words=3):
    from nltk import sent_tokenize
    _ensure_punkt()
    return [s.strip() for s in sent_tokenize(text) if len(s.split()) >= min_words]


def _prep_one(args):
    article, highlights, with_oracle, with_oracle_all = args
    from .extractive import encode, get_tokenizer
    from .metrics import greedy_oracle
    global _TOK
    if "_TOK" not in globals():
        _TOK = get_tokenizer()
    text = clean_article(article)
    sents = split_sentences(text)
    _, pos = encode(sents, _TOK)
    row = {"sentences": sents, "n_sent": len(sents), "n_fit": len(pos)}
    if with_oracle:
        row["oracle"] = greedy_oracle(sents[: len(pos)], highlights)
    if with_oracle_all:
        row["oracle_all"] = greedy_oracle(sents, highlights)
    return row


def preprocess(frame, name, with_oracle=True, with_oracle_all=False, workers=10):
    # Очистка, разбиение на предложения, число предложений в окне 512 токенов, разметка «оракула»
    path = C.CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(a, h, with_oracle, with_oracle_all) for a, h in zip(frame.article, frame.highlights)]
    with Pool(workers) as pool:
        rows = list(tqdm(pool.imap(_prep_one, jobs, chunksize=64), total=len(jobs), desc=name, mininterval=5))
    out = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out.to_parquet(path)
    return out


def lead(sentences, k=C.TOP_K):
    return "\n".join(list(sentences)[:k])


def oracle_summary(sentences, idx):
    s = list(sentences)
    return "\n".join(s[i] for i in idx)


# ---------- анализ ----------

def norm_hash(text):
    return hashlib.md5(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def length_stats(frame, tokenizer=None):
    st = pd.DataFrame({
        "article_words": frame.article.str.split().str.len(),
        "highlight_words": frame.highlights.str.split().str.len(),
        "highlight_sents": frame.highlights.str.count("\n") + 1,
    })
    if tokenizer is not None:
        st["article_tokens"] = [len(t) for t in tokenizer(frame.article.tolist(), add_special_tokens=False)["input_ids"]]
    return st


def sample(frame, n, seed=C.SEED):
    return frame.sample(n=min(n, len(frame)), random_state=seed).reset_index(drop=True)
