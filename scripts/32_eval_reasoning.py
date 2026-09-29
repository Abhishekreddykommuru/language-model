"""Stage 32: reasoning accuracy, pretrained against finetuned.

Both checkpoints are scored on the same held-out test set, two ways:

    generation   greedy decoding from the prompt; the text up to the first
                 sentence end must equal the gold answer exactly
    choice       each candidate answer ranked by length-normalised
                 log-likelihood, the best one taken

The second exists so the pretrained baseline means something. A model that has
never seen the answer marker scores zero on generation for reasons of output
format, which tells us nothing about whether it can compare two numbers.

Usage:
    python scripts/32_eval_reasoning.py --lang tel
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, ".")

from common import viz as V
from common.io.paths import ensure_drive_mounted, lang_dir, work_dir
from common.log import kv, section, table
from common.eval.generate import load_tokenizer
from common.reasoning.evaluate import (baselines, choice_accuracy,
                                       generation_accuracy)
from common.reasoning.generate import read_split
from scripts_util import check_tokenizer, load_model, run_name

MODELS = [("pretrained", ""), ("finetuned", "ft")]


def qualitative(detail_pre: list, detail_ft: list, n: int = 6) -> dict:
    """Examples worth reading: what finetuning fixed, and what it did not.

    Keyed by outcome rather than sampled at random, because an error analysis
    needs the cases that moved and the cases that are still wrong, not six
    arbitrary rows.
    """
    pre = {d["id"]: d for d in detail_pre}
    groups = {"fixed_by_finetuning": [], "still_wrong": [], "broken_by_finetuning": []}
    for d in detail_ft:
        was = pre.get(d["id"], {}).get("correct", False)
        if d["correct"] and not was:
            key = "fixed_by_finetuning"
        elif not d["correct"] and was:
            key = "broken_by_finetuning"
        elif not d["correct"]:
            key = "still_wrong"
        else:
            continue
        if len(groups[key]) < n:
            groups[key].append({
                "template": d["template"], "hops": d["hops"],
                "prompt": d["prompt"], "gold": d["gold"],
                "pretrained": pre.get(d["id"], {}).get("prediction", ""),
                "finetuned": d["prediction"]})
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["tel", "nep"])
    ap.add_argument("--work", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N examples, for a quick check")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default="report")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ensure_drive_mounted()
    work = args.work or work_dir()
    langdir = lang_dir(args.lang)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load(
        Path(f"{langdir}/configs/reasoning.yaml").read_text(encoding="utf-8"))
    sentence_end = cfg["sentence_end"]

    data_dir = Path(args.data_dir or f"{work}/{langdir}/9_reasoning")
    rows = read_split(str(data_dir / f"{args.split}.jsonl"))
    if args.limit:
        rows = rows[:args.limit]

    base = baselines(rows)
    section(f"Reasoning evaluation, {args.lang}")
    kv("split", f"{args.split}, {len(rows):,} examples")
    kv("random choice", f"{base['random_choice']:.1%}")
    kv("best constant answer", f"{base['majority_answer']:.1%}")
    kv("always the first entity named", f"{base['first_mentioned']:.1%}")

    results = {"lang": args.lang, "split": args.split, "n": len(rows),
               "baselines": base, "models": {}}
    details = {}
    for label, variant in MODELS:
        run = run_name(args.lang, variant)
        ckpt = f"{work}/{langdir}/7_checkpoints/{run}_{args.checkpoint}.pt"
        if not Path(ckpt).exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        model, mcfg, ck = load_model(ckpt, device)
        sp = load_tokenizer(langdir, args.lang)
        check_tokenizer(sp, mcfg, args.lang)

        gen = generation_accuracy(model, sp, rows, sentence_end, device,
                                  args.max_new_tokens, args.batch_size)
        cho = choice_accuracy(model, sp, rows, device, args.batch_size)
        details[label] = gen["predictions"]
        results["models"][label] = {
            "checkpoint": Path(ckpt).name, "step": ck["step"],
            "generation": {k: v for k, v in gen.items() if k != "predictions"},
            "choice": {k: v for k, v in cho.items() if k != "predictions"},
        }
        print(f"  {label}: generation {gen['accuracy']:.1%}, "
              f"choice {cho['accuracy']:.1%}", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\nexact match, greedy generation")
    families = sorted(results["models"]["finetuned"]["generation"]["by_template"])
    table(["family", "n", "pretrained", "finetuned", "change"],
          [[f,
            results["models"]["finetuned"]["generation"]["by_template"][f]["n"],
            f"{results['models']['pretrained']['generation']['by_template'][f]['accuracy']:.1%}",
            f"{results['models']['finetuned']['generation']['by_template'][f]['accuracy']:.1%}",
            f"{results['models']['finetuned']['generation']['by_template'][f]['accuracy'] - results['models']['pretrained']['generation']['by_template'][f]['accuracy']:+.1%}"]
           for f in families]
          + [["all", len(rows),
              f"{results['models']['pretrained']['generation']['accuracy']:.1%}",
              f"{results['models']['finetuned']['generation']['accuracy']:.1%}",
              f"{results['models']['finetuned']['generation']['accuracy'] - results['models']['pretrained']['generation']['accuracy']:+.1%}"]])

    print("\nlikelihood-ranked choice, same examples")
    table(["family", "pretrained", "finetuned"],
          [[f,
            f"{results['models']['pretrained']['choice']['by_template'][f]['accuracy']:.1%}",
            f"{results['models']['finetuned']['choice']['by_template'][f]['accuracy']:.1%}"]
           for f in families])

    print("\nby number of statements that must be combined")
    hops = sorted(results["models"]["finetuned"]["generation"]["by_hops"])
    table(["hops", "n", "pretrained", "finetuned"],
          [[h, results["models"]["finetuned"]["generation"]["by_hops"][h]["n"],
            f"{results['models']['pretrained']['generation']['by_hops'][h]['accuracy']:.1%}",
            f"{results['models']['finetuned']['generation']['by_hops'][h]['accuracy']:.1%}"]
           for h in hops])

    samples = qualitative(details["pretrained"], details["finetuned"])
    print("\nqualitative")
    for group, items in samples.items():
        kv(group.replace("_", " "), len(items))
    if samples["still_wrong"]:
        s = samples["still_wrong"][0]
        print(f"\n  a failure that survived finetuning [{s['template']}]")
        print(f"  prompt   : {s['prompt']}")
        print(f"  gold     : {s['gold']}")
        print(f"  finetuned: {s['finetuned']}")

    figs = Path(args.out) / "figures"
    V.plot_reasoning_accuracy(
        {f: {"pretrained": results["models"]["pretrained"]["generation"]["by_template"][f]["accuracy"],
             "finetuned": results["models"]["finetuned"]["generation"]["by_template"][f]["accuracy"]}
         for f in families},
        args.lang, str(figs / f"reasoning_accuracy_{args.lang}.png"),
        baseline=base["random_choice"])

    tabs = Path(args.out) / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    (tabs / f"reasoning_{args.lang}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (tabs / f"reasoning_samples_{args.lang}.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    (tabs / f"reasoning_predictions_{args.lang}.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    kv("metrics", tabs / f"reasoning_{args.lang}.json")
    kv("qualitative examples", tabs / f"reasoning_samples_{args.lang}.json")
    kv("accuracy plot", figs / f"reasoning_accuracy_{args.lang}.png")


if __name__ == "__main__":
    main()
