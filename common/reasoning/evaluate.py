"""Scoring a model on the synthetic reasoning test set.

Two measurements, because they answer different questions.

`generation_accuracy` is the headline number: the model
is given the prompt, decodes greedily, and its output up to the first sentence
end must equal the gold answer exactly. It is the honest end-to-end test, and it
is also unforgiving -- a pretrained model that has never seen the answer marker
will score near zero for reasons of format rather than reasoning.

`choice_accuracy` separates those two failures. The candidate answers are known
by construction, so each one is scored by its length-normalised log-likelihood
under the model and the best is taken. A model that cannot produce the format
can still show whether it prefers the right entity, which makes the pretrained
baseline interpretable instead of a flat zero.
"""
import math

import torch

from common.reasoning.data import IGNORE


def clean_answer(text: str, sentence_end: str) -> str:
    """The model's answer: everything up to the first sentence end or newline."""
    text = text.strip()
    for stop in (sentence_end, "\n", "?"):
        if stop and stop in text:
            text = text.split(stop)[0]
    return text.strip().strip(".।,;:").strip()


def exact_match(prediction: str, gold: str, sentence_end: str) -> bool:
    return clean_answer(prediction, sentence_end) == gold.strip()


def _bucket_by_length(rows: list, key) -> dict:
    """Group rows by tokenised prompt length.

    The model has no padding mask, so a batch can only hold prompts of one
    length: left-padding would put pad tokens inside the context every real
    position attends over.
    """
    buckets = {}
    for row in rows:
        buckets.setdefault(len(key(row)), []).append(row)
    return buckets


@torch.no_grad()
def generation_accuracy(model, sp, rows: list, sentence_end: str, device: str,
                        max_new_tokens: int = 16, batch_size: int = 32) -> dict:
    """Greedy-decode every prompt and compare with the gold answer."""
    model.eval()
    encoded = [(row, sp.encode(row["prompt"])) for row in rows]
    eos = sp.eos_id() if sp.eos_id() >= 0 else None
    predictions = {}
    for length, group in _bucket_by_length(encoded, lambda r: r[1]).items():
        if length + max_new_tokens > model.cfg.max_seq_len:
            for row, _ in group:
                predictions[row["id"]] = ""
            continue
        for i in range(0, len(group), batch_size):
            chunk = group[i:i + batch_size]
            idx = torch.tensor([ids for _, ids in chunk], dtype=torch.long,
                               device=device)
            out = model.generate(idx, max_new_tokens=max_new_tokens,
                                 temperature=0.0, eos_id=eos)
            for (row, _), seq in zip(chunk, out[:, length:].tolist()):
                ids = [t for t in seq
                       if 0 <= t < sp.get_piece_size() and t != eos]
                predictions[row["id"]] = sp.decode(ids)
    return _summarise(rows, predictions, sentence_end)


@torch.no_grad()
def choice_accuracy(model, sp, rows: list, device: str,
                    batch_size: int = 32) -> dict:
    """Rank each example's candidate answers by log-likelihood.

    Scores are divided by the number of answer tokens, so a candidate is not
    penalised for being a longer word -- without that, the shortest name wins
    far more often than it should.
    """
    model.eval()
    pad = sp.pad_id() if sp.pad_id() >= 0 else 0
    items = []
    for row in rows:
        prompt = sp.encode(row["prompt"])
        for choice in row["choices"]:
            full = sp.encode(f"{row['prompt']} {choice}")
            if len(full) <= len(prompt) or full[:len(prompt)] != prompt:
                continue
            items.append({"id": row["id"], "choice": choice,
                          "ids": full, "n_prompt": len(prompt)})

    scores = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        width = max(len(it["ids"]) for it in chunk)
        if width > model.cfg.max_seq_len:
            continue
        x = torch.full((len(chunk), width), pad, dtype=torch.long)
        y = torch.full((len(chunk), width), IGNORE, dtype=torch.long)
        for j, it in enumerate(chunk):
            ids = it["ids"]
            x[j, :len(ids)] = torch.tensor(ids)
            y[j, it["n_prompt"] - 1:len(ids) - 1] = torch.tensor(
                ids[it["n_prompt"]:])
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        logp = torch.log_softmax(logits.float(), dim=-1)
        keep = y != IGNORE
        gathered = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        total = (gathered * keep).sum(dim=-1)
        n = keep.sum(dim=-1).clamp_min(1)
        for j, it in enumerate(chunk):
            scores.setdefault(it["id"], []).append(
                (float(total[j] / n[j]), it["choice"]))

    predictions = {rid: max(cands)[1] for rid, cands in scores.items()}
    return _summarise(rows, predictions, sentence_end=None)


def _summarise(rows: list, predictions: dict, sentence_end) -> dict:
    """Overall and per-family accuracy, plus every prediction for inspection."""
    per_family, detail = {}, []
    correct = 0
    for row in rows:
        pred = predictions.get(row["id"], "")
        cleaned = clean_answer(pred, sentence_end) if sentence_end else pred
        hit = cleaned == row["answer"].strip()
        correct += hit
        fam = per_family.setdefault(row["template"], {"n": 0, "correct": 0})
        fam["n"] += 1
        fam["correct"] += hit
        detail.append({"id": row["id"], "template": row["template"],
                       "hops": row["hops"], "prompt": row["prompt"],
                       "gold": row["answer"], "prediction": cleaned,
                       "correct": bool(hit)})
    return {
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "by_template": {k: {"n": v["n"], "correct": v["correct"],
                            "accuracy": v["correct"] / max(v["n"], 1)}
                        for k, v in sorted(per_family.items())},
        "by_hops": _by_hops(detail),
        "predictions": detail,
    }


def _by_hops(detail: list) -> dict:
    out = {}
    for d in detail:
        row = out.setdefault(str(d["hops"]), {"n": 0, "correct": 0})
        row["n"] += 1
        row["correct"] += d["correct"]
    for row in out.values():
        row["accuracy"] = row["correct"] / max(row["n"], 1)
    return dict(sorted(out.items()))


def baselines(rows: list) -> dict:
    """What a model gets without reasoning at all.

    `random_choice` is the expected accuracy of guessing uniformly among the
    candidates; `majority_answer` is the best single constant answer; and
    `first_mentioned` is the accuracy of naming whichever entity appears first
    in the prompt, which is the shortcut a template-matching model would find if
    statement order carried the answer. It should sit at chance, because the
    generator shuffles the statements.
    """
    if not rows:
        return {}
    random_acc = sum(1 / len(r["choices"]) for r in rows) / len(rows)
    counts = {}
    for r in rows:
        counts[r["answer"]] = counts.get(r["answer"], 0) + 1
    first = 0
    for r in rows:
        positions = [(r["prompt"].find(e), e) for e in r["choices"]
                     if r["prompt"].find(e) >= 0]
        if positions and min(positions)[1] == r["answer"]:
            first += 1
    return {
        "random_choice": random_acc,
        "majority_answer": max(counts.values()) / len(rows),
        "first_mentioned": first / len(rows),
    }


def perplexity_on_answers(model, dataset, batch_size: int, device: str) -> float:
    """Mean cross-entropy over answer tokens only, as a training-free check."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in dataset.sequential_batches(batch_size, device):
            _, loss, _ = model(x, y)
            total += loss.item()
            n += 1
    return math.exp(min(total / max(n, 1), 20))
