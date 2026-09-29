"""Train all three algorithms, sweep vocabulary sizes, pick the winner.

Vocabulary size is chosen from fertility and unknown-token rate on held-out
text, so the sweep is a deliverable in its own right. The tokenizer sees the
train split only: fitting a vocabulary to held-out text is a quiet form of
leakage.

Usage:
    python scripts/08_train_tokenizers.py --lang tel
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, ".")

from common.io.paths import ensure_drive_mounted, lang_dir, scratch_dir, work_dir
from common.io.shards import read_dir
from common.tokenizer.evaluate import (PROBE_WORDS, TokenizerWrapper, evaluate,
                                       phase2_param_estimate, segmentation_examples,
                                       select_best)
from common.log import kv, section, table
from common.tokenizer.train import export_vocab_txt, train

MIN_LINE_CHARS = 20

#: Lines the from-scratch implementations are validated on. They are pure Python
#: and about two orders of magnitude slower than the libraries, so they run on a
#: sample to demonstrate the algorithms rather than at full scale.
SCRATCH_VALIDATION_LINES = 20_000
SCRATCH_VALIDATION_VOCAB = 2_000


def load_algo_config(langdir: str, algorithm: str) -> dict:
    """Load the per-algorithm hyperparameter file, if one exists."""
    path = Path(f"{langdir}/configs/tokenizer_{algorithm}.yaml")
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("hyperparameters", {})


def validate_against_scratch(lines: list, algorithms: list, algo_cfgs: dict,
                             common_cfg: dict, out_path: str) -> list:
    """Train our own implementations on a sample and compare with the libraries.

    Same corpus, same vocabulary size. Exact agreement is not expected, since
    tie-breaking and normalisation differ, but the compression figures should
    land close together.
    """
    from common.tokenizer.scratch import train_scratch

    sample = lines[:SCRATCH_VALIDATION_LINES]
    rows = []
    for algo in algorithms:
        print(f"  {algo}, from scratch, on {len(sample):,} lines", flush=True)
        try:
            cfg = dict(common_cfg)
            cfg.update(algo_cfgs.get(algo, {}))
            t0 = time.time()
            model = train_scratch(algo, sample, SCRATCH_VALIDATION_VOCAB, cfg,
                                  f"{out_path}/scratch_{algo}.json", verbose=False)
            n_tok = sum(len(model.encode(l)) for l in sample[:500])
            n_chars = sum(len(l) for l in sample[:500])
            rows.append({
                "algorithm": algo,
                "engine": "from_scratch",
                "vocab_size": len(model.vocab),
                "chars_per_token": n_chars / max(n_tok, 1),
                "num_merges": getattr(model, "num_merges", None),
                "min_frequency": getattr(model, "min_frequency", None),
                "train_seconds": round(time.time() - t0, 1),
            })
            print(f"    vocab {len(model.vocab):,}, "
                  f"{rows[-1]['chars_per_token']:.3f} chars/token, "
                  f"{rows[-1]['train_seconds']}s", flush=True)
        except Exception as exc:
            print(f"    {algo} failed, {exc}", flush=True)
    return rows


def dump_text(shard_dir: str, out_txt: str, max_lines: int) -> int:
    """Write one line per sentence for the tokenizer trainer.

    An existing dump holding enough lines is reused: reading the corpus back
    off the Drive mount is the slowest part of this stage when Drive throttles,
    and the dump is deterministic.
    """
    existing = Path(out_txt)
    if existing.exists():
        with open(existing, encoding="utf-8") as fh:
            have = sum(1 for _ in fh)
        if have >= max_lines:
            print(f"  reusing {out_txt} ({have:,} lines already dumped)",
                  flush=True)
            return have
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_txt, "w", encoding="utf-8") as fh:
        for d in read_dir(shard_dir):
            for line in d.get("text", "").split("\n"):
                line = line.strip()
                if len(line) < MIN_LINE_CHARS:
                    continue
                fh.write(line + "\n")
                n += 1
                if n >= max_lines:
                    return n
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--skip-scratch-validation", action="store_true",
                    help="skip the pure-Python reference implementations")
    ap.add_argument("--algorithms", default=None,
                    help="comma-separated override of the swept algorithms")
    ap.add_argument("--vocab-sizes", default=None,
                    help="comma-separated override of the swept vocabulary sizes")
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    cfg = yaml.safe_load(open(f"{lang_dir(args.lang)}/configs/tokenizer.yaml",
                              encoding="utf-8"))
    common_cfg, sweep_cfg = cfg["common"], cfg["sweep"]
    if args.algorithms:
        sweep_cfg["algorithms"] = [a.strip() for a in args.algorithms.split(",")]
    if args.vocab_sizes:
        sweep_cfg["vocab_sizes"] = [int(v) for v in args.vocab_sizes.split(",")]
    sel = cfg.get("selection", {})
    langdir = lang_dir(args.lang)
    algo_cfgs = {a: load_algo_config(langdir, a) for a in sweep_cfg["algorithms"]}
    tokdir = f"{work}/{lang_dir(args.lang)}/tokenizer"
    Path(tokdir).mkdir(parents=True, exist_ok=True)

    train_txt = f"{scratch_dir()}/tok_train_{args.lang}.txt"
    held_txt = f"{scratch_dir()}/tok_held_{args.lang}.txt"
    n_train = dump_text(f"{work}/{lang_dir(args.lang)}/5_final/train", train_txt,
                        sweep_cfg["sweep_sample_lines"])
    n_held = dump_text(f"{work}/{lang_dir(args.lang)}/5_final/val", held_txt, 50_000)
    held_lines = [l.rstrip("\n") for l in open(held_txt, encoding="utf-8")]

    if n_train == 0:
        raise SystemExit(
            f"No text to train on: {work}/{lang_dir(args.lang)}/5_final/train "
            f"is empty.\n"
            f"  Stages 05 to 07 must succeed first. Run them one at a time and "
            f"read the error:\n"
            f"    python scripts/05_clean.py --lang {args.lang} --split train\n"
            f"    python scripts/06_dedup.py --lang {args.lang}\n"
            f"    python scripts/07_finalize.py --lang {args.lang}")

    section(f"Tokenizer sweep, {args.lang}")
    kv("training lines", f"{n_train:,}")
    kv("held-out lines", f"{n_held:,}")
    kv("configurations", f"{len(sweep_cfg['algorithms'])} algorithms x "
                        f"{len(sweep_cfg['vocab_sizes'])} vocabulary sizes")
    print()

    results = []
    for algo in sweep_cfg["algorithms"]:
        for vocab in sweep_cfg["vocab_sizes"]:
            tag = f"{args.lang}_{algo}_{vocab}"
            print(f"  training {algo} at vocab {vocab:,}", flush=True)
            try:
                merged_cfg = dict(common_cfg)
                merged_cfg.update(algo_cfgs.get(algo, {}))
                model = train(algo, train_txt, f"{tokdir}/sweep/{tag}", vocab,
                              merged_cfg)
                metrics = evaluate(TokenizerWrapper(model, algo), held_lines)
                metrics.pop("top_50", None)
                metrics.update({"algorithm": algo, "vocab_size": vocab,
                                "phase2": phase2_param_estimate(vocab)})
                results.append(metrics)
            except Exception as exc:
                print(f"    failed, {exc}", flush=True)

    if not results:
        raise SystemExit("every sweep configuration failed, see the log above")

    with open(f"{tokdir}/sweep_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    section(f"Sweep results, {args.lang}")
    table(["algorithm", "vocab", "chars/token", "tokens/word", "unk rate",
           "byte fallback", "embedding share"],
          [[r["algorithm"], f"{r['vocab_size']:,}",
            f"{r['chars_per_token']:.3f}", f"{r['tokens_per_word']:.3f}",
            f"{r['unk_rate']:.5f}", f"{r['byte_fallback_rate']:.5f}",
            f"{r['phase2']['embedding_share']:.1%}"] for r in results])

    chosen = select_best(
        results,
        max_embedding_share=sel.get("max_embedding_share", 0.50),
        within_frac=sel.get("within_frac", 0.95),
        max_vocab=sel.get("max_vocab", 65536),
    )
    section(f"Chosen tokenizer, {args.lang}")
    kv("algorithm", chosen["algorithm"])
    kv("vocabulary size", f"{chosen['vocab_size']:,}")
    kv("chars per token", f"{chosen['chars_per_token']:.3f}")
    kv("Phase 2 embedding share", f"{chosen['phase2']['embedding_share']:.1%}")

    n_full = dump_text(f"{work}/{lang_dir(args.lang)}/5_final/train", train_txt,
                       sweep_cfg["final_sample_lines"])
    print(f"\n  retraining the winner on {n_full:,} lines", flush=True)
    model = train(chosen["algorithm"], train_txt, f"{tokdir}/{args.lang}",
                  chosen["vocab_size"], common_cfg)
    export_vocab_txt(model, f"{tokdir}/{args.lang}.vocab.txt", chosen["algorithm"])

    wrapper = TokenizerWrapper(model, chosen["algorithm"])
    final_metrics = evaluate(wrapper, held_lines)
    examples = segmentation_examples(wrapper, PROBE_WORDS[args.lang])
    with open(f"{tokdir}/chosen.json", "w") as fh:
        json.dump({"chosen": chosen, "model": model,
                   "final_metrics": {k: v for k, v in final_metrics.items()
                                     if k != "top_50"},
                   "examples": examples}, fh, indent=2, ensure_ascii=False)

    if not args.skip_scratch_validation:
        section("Cross-check against the from-scratch implementations")
        scratch_rows = validate_against_scratch(
            held_lines + [l.rstrip("\n") for l in open(train_txt, encoding="utf-8")][:SCRATCH_VALIDATION_LINES],
            sweep_cfg["algorithms"], algo_cfgs, common_cfg, tokdir)
        with open(f"{tokdir}/scratch_validation.json", "w") as fh:
            json.dump(scratch_rows, fh, indent=2)

    section(f"Final tokenizer, {args.lang}")
    kv("model", model)
    kv("vocabulary", f"{tokdir}/{args.lang}.vocab.txt")
    print("\nhow it segments the probe words")
    table(["word", "pieces", "count"],
          [[ex["word"], " ".join(ex["pieces"]), ex["n_pieces"]]
           for ex in examples[:10]], aligns="llr")


if __name__ == "__main__":
    main()
