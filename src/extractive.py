import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config as C
from .metrics import rouge


def get_tokenizer():
    return AutoTokenizer.from_pretrained(C.ROBERTA)


def encode(sentences, tokenizer, max_len=C.MAX_TOKENS):
    # <s> предложение </s> <s> предложение </s> ... — вектор <s> представляет предложение
    ids, cls_pos = [], []
    for s in sentences:
        t = tokenizer.encode(s, add_special_tokens=False)
        room = max_len - len(ids)
        if room < 2 + min(len(t), 8):
            break
        piece = [tokenizer.cls_token_id] + t[: room - 2] + [tokenizer.sep_token_id]
        cls_pos.append(len(ids))
        ids += piece
    return ids, cls_pos


class ExtDataset(Dataset):
    def __init__(self, frame, tokenizer, with_labels=True):
        self.sents = frame.sentences.tolist()
        self.labels = frame.oracle.tolist() if with_labels else None
        self.tok = tokenizer

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        ids, pos = encode(list(self.sents[i]), self.tok)
        y = np.zeros(len(pos), np.float32)
        if self.labels is not None:
            for j in self.labels[i]:
                if j < len(pos):
                    y[j] = 1.0
        return ids, pos, y


def collate(batch, pad_id=1):
    L = max(len(b[0]) for b in batch)
    S = max(len(b[1]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    pos = torch.zeros((len(batch), S), dtype=torch.long)
    smask = torch.zeros((len(batch), S), dtype=torch.bool)
    y = torch.zeros((len(batch), S))
    for k, (i, p, l) in enumerate(batch):
        ids[k, : len(i)] = torch.tensor(i)
        att[k, : len(i)] = 1
        pos[k, : len(p)] = torch.tensor(p)
        smask[k, : len(p)] = True
        y[k, : len(l)] = torch.from_numpy(l)
    return ids, att, pos, smask, y


class ExtSummarizer(torch.nn.Module):
    def __init__(self, name=C.ROBERTA):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name, add_pooling_layer=False)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, att, pos):
        h = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        sent = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        return self.head(sent).squeeze(-1)


def _trigrams(text):
    w = text.lower().split()
    return set(zip(w, w[1:], w[2:]))


def select(sentences, scores, k=C.TOP_K):
    # Top-k предложений с блокировкой повторяющихся триграмм, в порядке следования в статье
    chosen, seen = [], set()
    for j in np.argsort(-scores):
        tri = _trigrams(sentences[j])
        if tri & seen:
            continue
        chosen.append(int(j))
        seen |= tri
        if len(chosen) == k:
            break
    return "\n".join(sentences[j] for j in sorted(chosen))


def make_loader(frame, tokenizer, batch_size, shuffle, workers=4, with_labels=True, seed=C.SEED):
    return DataLoader(ExtDataset(frame, tokenizer, with_labels), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def predict(model, frame, tokenizer, batch_size=32, workers=4, device="cuda"):
    model.eval().to(device)
    loader = make_loader(frame, tokenizer, batch_size, shuffle=False, workers=workers, with_labels=False)
    sents = frame.sentences.tolist()
    out, k = [], 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ids, att, pos, smask, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids.to(device), att.to(device), pos.to(device)).float()
        logits = logits.masked_fill(~smask.to(device), -1e9).cpu().numpy()
        for row, m in zip(logits, smask.numpy()):
            out.append(select(list(sents[k]), row[: m.sum()]))
            k += 1
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / len(frame)


CPU_THREADS = torch.get_num_threads()


@torch.no_grad()
def latency_ms(model, frame, tokenizer, device, n=30, warmup=3):
    # Полная обработка одной статьи: токенизация, прогон модели, выбор предложений (батч 1)
    prev = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    model.eval().to(device)
    times = []
    try:
        for k, sents in enumerate(frame.sentences.tolist()[: n + warmup]):
            t0 = time.perf_counter()
            ids, pos = encode(list(sents), tokenizer)
            ids_t = torch.tensor([ids], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(ids_t, torch.ones_like(ids_t), torch.tensor([pos], device=device)).float()
            select(list(sents), logits[0].cpu().numpy())
            if device == "cuda":
                torch.cuda.synchronize()
            if k >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(prev)
    return float(np.median(times))


def lr_lambda(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def train(model, run_name, train_frame, val_frame, tokenizer, epochs, lr, batch_size=16,
          weight_decay=0.01, warmup_frac=0.1, evals_per_epoch=2, workers=4, device="cuda"):
    ckpt_dir, out_dir = C.CKPT_DIR / run_name, C.OUT_DIR / "runs" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    model.to(device)
    loader = make_loader(train_frame, tokenizer, batch_size, shuffle=True, workers=workers)
    steps_per_epoch = len(loader)
    total = epochs * steps_per_epoch
    eval_at = {round(steps_per_epoch * (e + (q + 1) / evals_per_epoch)) for e in range(epochs) for q in range(evals_per_epoch)}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, int(warmup_frac * total)))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    history, best, step, run_loss, n_loss = [], -1.0, 0, 0.0, 0
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    bar = tqdm(total=total, desc=run_name, mininterval=5)
    for epoch in range(epochs):
        model.train()
        for ids, att, pos, smask, y in loader:
            ids, att, pos, smask, y = (t.to(device, non_blocking=True) for t in (ids, att, pos, smask, y))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, att, pos)
            loss = (bce(logits.float(), y) * smask).sum() / smask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            n_loss += 1
            bar.update(1)
            if step in eval_at:
                preds, _ = predict(model, val_frame, tokenizer, workers=workers, device=device)
                r = rouge(preds, val_frame.highlights.tolist()).mean()
                row = {"step": step, "epoch": step / steps_per_epoch, "train_loss": run_loss / n_loss,
                       **{f"val_{k}": v for k, v in r.items()}, "val_ROUGE_mean": r.mean(),
                       "elapsed_min": (time.time() - t_start) / 60}
                history.append(row)
                run_loss, n_loss = 0.0, 0
                if row["val_ROUGE_mean"] > best:
                    best = row["val_ROUGE_mean"]
                    torch.save(model.state_dict(), ckpt_dir / "best.pt")
                pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
                bar.write(f"step {step}/{total} | train_loss {row['train_loss']:.4f} | val R-1 {r['ROUGE-1']:.2f} "
                          f"R-2 {r['ROUGE-2']:.2f} R-L {r['ROUGE-L']:.2f} | {row['elapsed_min']:.1f} min")
                model.train()
    bar.close()
    info = {"run": run_name, "epochs": epochs, "lr": lr, "batch_size": batch_size, "train_size": len(train_frame),
            "steps": total, "train_min": (time.time() - t_start) / 60,
            "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30, "best_val_ROUGE_mean": best}
    (out_dir / "params.json").write_text(json.dumps(info, indent=2))
    return pd.DataFrame(history), info


def load_best(run_name, device="cuda"):
    model = ExtSummarizer()
    model.load_state_dict(torch.load(C.CKPT_DIR / run_name / "best.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval()


def finished(run_name):
    return (C.OUT_DIR / "runs" / run_name / "params.json").exists() and (C.CKPT_DIR / run_name / "best.pt").exists()
