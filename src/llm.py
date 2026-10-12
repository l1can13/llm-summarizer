import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI
from tqdm import tqdm

from . import config as C

PROMPTS = {
    "basic": (
        "Summarize the following news article in three sentences. "
        "Output only the summary.\n\nArticle:\n{article}"
    ),
    "highlights": (
        "Write the highlights of the news article below: {n_sent} short sentences, about {n_words} words "
        "in total, stating the key facts (who, what, where, when). Use only information from the article. "
        "Output only the highlights, one sentence per line, without numbering or introduction."
        "\n\nArticle:\n{article}"
    ),
}


def truncate_words(text, max_words=C.LLM_MAX_ARTICLE_WORDS):
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words])


def build_prompt(name, article, **kw):
    return PROMPTS[name].format(article=truncate_words(article), **kw)


_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_output(text):
    text = _THINK.sub("", text or "").strip()
    text = re.sub(r"^(here (is|are)[^:\n]*:|summary:|highlights:)\s*", "", text, flags=re.I).strip()
    lines = [re.sub(r"^\s*([-*•]|\d+[.)])\s*", "", l).strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


# ---------- клиенты ----------

def lmstudio_client():
    return OpenAI(base_url=C.LMSTUDIO_URL, api_key="lm-studio", max_retries=3, timeout=600)


def openai_client():
    key = dotenv_values(C.LAB_DIR / ".env").get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Не задан OPENAI_API_KEY: впишите ключ в lr2/.env")
    return OpenAI(api_key=key, max_retries=6, timeout=120)


def generate(client, model, prompt, params):
    t = time.perf_counter()
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **params)
    latency = time.perf_counter() - t
    u = r.usage
    return {
        "raw": r.choices[0].message.content or "",
        "finish_reason": r.choices[0].finish_reason,
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "latency_s": latency,
    }


def run(run_name, frame, client, model, prompt_name, params, prompt_kw=None, workers=1):
    # Ответы пишутся построчно в JSONL: при повторном запуске уже готовые статьи пропускаются
    path = C.GEN_DIR / f"{run_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = frame[~frame.id.isin(done)]
    lock = threading.Lock()

    def one(row):
        res = generate(client, model, build_prompt(prompt_name, row.article, **(prompt_kw or {})), params)
        res.update(id=row.id, summary=clean_output(res["raw"]))
        with lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    if len(todo):
        print(f"{run_name}: готово {len(done)}, осталось {len(todo)}")
        with ThreadPoolExecutor(workers) as pool:
            list(tqdm(pool.map(one, todo.itertuples()), total=len(todo), desc=run_name, mininterval=5))
    return load_run(run_name, frame)


def load_run(run_name, frame):
    rows = [json.loads(l) for l in (C.GEN_DIR / f"{run_name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    res = pd.DataFrame(rows).drop_duplicates("id", keep="last").set_index("id")
    return res.loc[frame.id].reset_index()


def openai_cost(gen):
    pin, pout = C.OPENAI_PRICE
    return (gen.prompt_tokens.sum() * pin + gen.completion_tokens.sum() * pout) / 1e6


# ---------- LM Studio ----------

LMS = shutil.which("lms") or str(Path.home() / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms"))


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def lms(*args):
    r = subprocess.run([LMS, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = _ANSI.sub("", (r.stdout + r.stderr).replace("\r", "\n"))
    return "\n".join(l for l in out.splitlines() if l.strip() and not l.startswith("Loading ")).strip()


def lmstudio_up(model=C.QWEN_MODEL, context=4096):
    lms("server", "start")
    if model not in lms("ps"):
        print(lms("load", model, "-y", "-c", str(context)))
    return lms("ps")


def lmstudio_down():
    return lms("unload", "--all")
