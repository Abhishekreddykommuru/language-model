"""Stage 22: generation quality.

Continues held-out prefixes with greedy decoding and with temperatures 0.5, 1.0
and 1.5, then scores each setting against the true continuation with BLEU,
chrF++ and ROUGE-L, plus repetition and Distinct-1/2.

Usage:
    python scripts/22_generate.py --lang tel
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section, table
from common.eval.generate import (decode_all, generate_batch, held_out_prefixes,
                                  load_tokenizer)
from common.eval.textmetrics import all_metrics
from common.train.data import load_splits
from scripts_util import load_model, run_name

SETTINGS = [("greedy", 0.0), ("t0.5", 0.5), ("t1.0", 1.0), ("t1.5", 1.5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--prefix-len", type=int, default=64)
    ap.add_argument("--continuation-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="report")
    ap.add_argument("--device", default=None)
    ap.add_argument("--variant", default="",
                    help="evaluate a non-standard run, e.g. ft or nopos")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run = run_name(args.lang, args.variant)
    ckpt = f"{work}/{langdir}/7_checkpoints/{run}_{args.checkpoint}.pt"
    if not Path(ckpt).exists():
        raise SystemExit(f"no checkpoint at {ckpt}; run scripts/20_train.py first")

    model, mcfg, ck = load_model(ckpt, device)
    splits = load_splits(work, langdir, mcfg.max_seq_len)
    held = splits.get("test") or splits["val"]
    sp = load_tokenizer(langdir, args.lang)

    pairs = held_out_prefixes(held, args.n_prompts, args.prefix_len,
                              args.continuation_len)
    prefixes = [p for p, _ in pairs]
    references = decode_all(sp, [r for _, r in pairs])
    prompts_text = decode_all(sp, prefixes)

    section(f"Generation, {run}")
    kv("checkpoint", f"step {ck['step']:,}")
    kv("prompts", f"{args.n_prompts} prefixes of {args.prefix_len} tokens")
    kv("continuation length", f"{args.continuation_len} tokens")

    results, samples, rows = {}, {}, []
    for name, temp in SETTINGS:
        outs = []
        for i in range(0, len(prefixes), args.batch_size):
            chunk = prefixes[i:i + args.batch_size]
            outs.extend(generate_batch(model, chunk, args.continuation_len,
                                       temp, device))
        texts = decode_all(sp, outs)
        m = all_metrics(texts, references)
        results[name] = {"temperature": temp, **m}
        samples[name] = [
            {"prompt": prompts_text[i], "generated": texts[i],
             "reference": references[i]} for i in range(min(5, len(texts)))]
        rows.append([name, f"{m['bleu']:.2f}", f"{m['chrf_pp']:.2f}",
                     f"{m['rouge_l_f']:.4f}", f"{m['distinct_1']:.3f}",
                     f"{m['distinct_2']:.3f}", f"{m['repetition_4gram']:.3f}"])
        print(f"  {name} done", flush=True)

    print()
    table(["decoding", "BLEU-4", "chrF++", "ROUGE-L", "distinct-1",
           "distinct-2", "4-gram repetition"], rows)

    print("\nsample continuation, greedy")
    s = samples["greedy"][0]
    print(f"  prompt   : {s['prompt'][:160]}")
    print(f"  generated: {s['generated'][:160]}")
    print(f"  reference: {s['reference'][:160]}")

    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"generation_{run}.json").write_text(
        json.dumps({"lang": args.lang, "run": run, "step": ck["step"],
                    "settings": results}, indent=2, ensure_ascii=False))
    (out / f"samples_{run}.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False))
    print()
    kv("metrics", out / f"generation_{run}.json")
    kv("samples", out / f"samples_{run}.json")


if __name__ == "__main__":
    main()
