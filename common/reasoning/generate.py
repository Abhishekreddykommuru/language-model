"""Synthetic comparative-reasoning examples, built from a per-language config.

Every example is generated programmatically, so the ground-truth answer is
known by construction rather than annotated: the generator draws the numbers or
the ordering first and renders the sentences from it. Nothing here is
downloaded, translated or model-written.

Five template families, in increasing order of how much chaining they need:

    two_way_value    two stated values, ask which is more or less
    three_way_value  three stated values, ask for the maximum or minimum
    transitive_3     A > B, B > C, no numbers; ask which is largest/smallest
    equality         two values, equal or not, answer yes or no
    transitive_4     A > B > C > D as three separate statements

Statements are shuffled inside every example, so the answer can never be read
off the surface order -- a model that always answers with the first-mentioned
entity scores at chance.

Leakage is controlled in two independent ways:

    1. Entity pools are disjoint across train, validation and test. No name the
       model saw during finetuning appears at evaluation.
    2. `transitive_4` is generated for the test split only. It is a relation
       pattern the model never sees, so accuracy on it separates a model that
       learned to chain inequalities from one that memorised the three-term
       surface form.
"""
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")


@dataclass
class Example:
    """One generated item, with everything needed to score it later."""

    id: str
    split: str
    template: str
    attribute: str
    pool: str
    entities: list
    values: list
    facts: list
    question: str
    answer: str
    #: Candidate answers, for likelihood-ranked scoring of a model that cannot
    #: yet produce the output format on its own.
    choices: list
    prompt: str = ""
    text: str = ""
    #: How many statements have to be combined to answer. 1 means the answer is
    #: readable from a single sentence; 3 means three inequalities chain.
    hops: int = 1

    def render(self, marker: str, sentence_end: str):
        self.prompt = " ".join(self.facts + [self.question, marker])
        self.text = f"{self.prompt} {self.answer}{sentence_end}"


class Generator:
    """Draws examples for one language from its reasoning config."""

    def __init__(self, config_path: str):
        self.cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.pat = self.cfg["patterns"]
        self.marker = self.cfg["answer_marker"]
        self.end = self.cfg["sentence_end"]
        self.attributes = self.cfg["attributes"]
        self.counts = {t["id"]: t for t in self.cfg["templates"]}
        self.rng = random.Random(self.cfg.get("seed", 0))

    # ---------------- helpers ----------------

    def pool_of(self, split: str, pool: str) -> list:
        return self.cfg["entities"][pool][split]

    def words(self, pool: str) -> dict:
        p = self.cfg["entities"][pool]
        return {"q": p["question_word"], "all": p["all_word"],
                "both": p["both_word"]}

    def pick_attribute(self, split: str, n_entities: int) -> dict:
        """An attribute whose pool has enough distinct entities for this split.

        The object pools are smaller than the people pools, so a four-entity
        template can only use an attribute whose pool is large enough.
        """
        usable = [a for a in self.attributes
                  if len(self.pool_of(split, a["pool"])) >= n_entities]
        return self.rng.choice(usable)

    def distinct_values(self, attr: dict, n: int) -> list:
        """n distinct values from the attribute's range, in random order.

        Distinctness matters: two entities sharing a value would make a
        "which is more" question ambiguous and its gold answer wrong.
        """
        span = attr["high"] - attr["low"] + 1
        return self.rng.sample(range(attr["low"], attr["low"] + span), n)

    def fact(self, entity: str, attr: dict, value: int) -> str:
        return self.pat["fact"].format(a=entity, noun=attr["noun"], v=value,
                                       unit=attr["unit"])

    def relation(self, high: str, low: str, attr: dict) -> str:
        return self.pat["relation_more"].format(a=high, b=low,
                                                noun=attr["noun"])

    def question(self, kind: str, attr: dict, pool: str) -> str:
        return self.pat[f"question_{kind}"].format(noun=attr["noun"],
                                                   **self.words(pool))

    # ---------------- template families ----------------

    def two_way_value(self, split: str) -> Example:
        attr = self.pick_attribute(split, 2)
        a, b = self.rng.sample(self.pool_of(split, attr["pool"]), 2)
        va, vb = self.distinct_values(attr, 2)
        facts = [self.fact(a, attr, va), self.fact(b, attr, vb)]
        self.rng.shuffle(facts)
        kind = self.rng.choice(["more", "less"])
        answer = (a if va > vb else b) if kind == "more" else (a if va < vb else b)
        return Example(
            id="", split=split, template="two_way_value",
            attribute=attr["key"], pool=attr["pool"], entities=[a, b],
            values=[va, vb], facts=facts,
            question=self.question(kind, attr, attr["pool"]),
            answer=answer, choices=[a, b], hops=1)

    def three_way_value(self, split: str) -> Example:
        attr = self.pick_attribute(split, 3)
        ents = self.rng.sample(self.pool_of(split, attr["pool"]), 3)
        vals = self.distinct_values(attr, 3)
        facts = [self.fact(e, attr, v) for e, v in zip(ents, vals)]
        self.rng.shuffle(facts)
        kind = self.rng.choice(["max", "min"])
        pick = max if kind == "max" else min
        answer = ents[vals.index(pick(vals))]
        return Example(
            id="", split=split, template="three_way_value",
            attribute=attr["key"], pool=attr["pool"], entities=ents,
            values=vals, facts=facts,
            question=self.question(kind, attr, attr["pool"]),
            answer=answer, choices=list(ents), hops=2)

    def _chain(self, split: str, n: int, template: str) -> Example:
        """n entities in a strict order, stated as n-1 pairwise inequalities.

        No numbers appear, so the only route to the answer is composing the
        statements. `ordered` runs from largest to smallest.
        """
        attr = self.pick_attribute(split, n)
        ordered = self.rng.sample(self.pool_of(split, attr["pool"]), n)
        facts = [self.relation(ordered[i], ordered[i + 1], attr)
                 for i in range(n - 1)]
        self.rng.shuffle(facts)
        kind = self.rng.choice(["max", "min"])
        answer = ordered[0] if kind == "max" else ordered[-1]
        return Example(
            id="", split=split, template=template, attribute=attr["key"],
            pool=attr["pool"], entities=list(ordered), values=[], facts=facts,
            question=self.question(kind, attr, attr["pool"]),
            answer=answer, choices=list(ordered), hops=n - 1)

    def transitive_3(self, split: str) -> Example:
        return self._chain(split, 3, "transitive_3")

    def transitive_4(self, split: str) -> Example:
        return self._chain(split, 4, "transitive_4")

    def equality(self, split: str) -> Example:
        attr = self.pick_attribute(split, 2)
        a, b = self.rng.sample(self.pool_of(split, attr["pool"]), 2)
        equal = self.rng.random() < 0.5
        if equal:
            va = vb = self.distinct_values(attr, 1)[0]
        else:
            va, vb = self.distinct_values(attr, 2)
        facts = [self.fact(a, attr, va), self.fact(b, attr, vb)]
        self.rng.shuffle(facts)
        return Example(
            id="", split=split, template="equality", attribute=attr["key"],
            pool=attr["pool"], entities=[a, b], values=[va, vb], facts=facts,
            question=self.question("equal", attr, attr["pool"]),
            answer=self.pat["answer_yes"] if equal else self.pat["answer_no"],
            choices=[self.pat["answer_yes"], self.pat["answer_no"]], hops=1)

    # ---------------- driver ----------------

    def build(self, split: str) -> list:
        """Every example for one split, deduplicated by prompt text.

        Small pools plus short templates make collisions likely -- there are
        only so many ways to compare two of eight objects -- so a repeated
        prompt is redrawn rather than kept. The retry budget is generous but
        finite; if a family cannot fill its quota the shortfall is reported
        instead of looping forever.
        """
        families = {
            "two_way_value": self.two_way_value,
            "three_way_value": self.three_way_value,
            "transitive_3": self.transitive_3,
            "equality": self.equality,
            "transitive_4": self.transitive_4,
        }
        seen, out, shortfall = set(), [], {}
        for tid, make in families.items():
            want = self.counts.get(tid, {}).get(split, 0)
            got, attempts, budget = 0, 0, want * 50 + 1000
            while got < want and attempts < budget:
                attempts += 1
                ex = make(split)
                ex.render(self.marker, self.end)
                if ex.prompt in seen:
                    continue
                seen.add(ex.prompt)
                ex.id = f"{split}-{tid}-{got:05d}"
                out.append(ex)
                got += 1
            if got < want:
                shortfall[tid] = want - got
        self.rng.shuffle(out)
        self.last_shortfall = shortfall
        return out


def write_split(examples: list, path: str):
    """One JSON object per line, in generation order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def read_split(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def leakage_report(splits: dict, cfg: dict) -> dict:
    """Evidence that train, validation and test do not overlap.

    Checks both controls directly rather than trusting the construction:
    entity vocabularies must not intersect, prompts must not repeat, and the
    held-out family must be absent from training.
    """
    entities, prompts, families = {}, {}, {}
    for name, rows in splits.items():
        entities[name] = {e for r in rows for e in r["entities"]}
        prompts[name] = {r["prompt"] for r in rows}
        families[name] = sorted({r["template"] for r in rows})

    def overlap(a, b, field):
        return sorted(field[a] & field[b])

    return {
        "entity_overlap": {
            "train_val": overlap("train", "val", entities),
            "train_test": overlap("train", "test", entities),
            "val_test": overlap("val", "test", entities),
        },
        "prompt_overlap_counts": {
            "train_val": len(prompts["train"] & prompts["val"]),
            "train_test": len(prompts["train"] & prompts["test"]),
            "val_test": len(prompts["val"] & prompts["test"]),
        },
        "families_per_split": families,
        "held_out_families": sorted(set(families["test"])
                                    - set(families["train"])),
        "entities_per_split": {k: len(v) for k, v in entities.items()},
    }


def dataset_card(splits: dict, cfg: dict, leakage: dict) -> dict:
    """Sizes, template variety and answer balance, for the report."""
    per_split = {}
    for name, rows in splits.items():
        by_template, by_attribute, by_answer, by_hops = {}, {}, {}, {}
        lengths = []
        for r in rows:
            by_template[r["template"]] = by_template.get(r["template"], 0) + 1
            by_attribute[r["attribute"]] = by_attribute.get(r["attribute"], 0) + 1
            by_answer[r["answer"]] = by_answer.get(r["answer"], 0) + 1
            by_hops[r["hops"]] = by_hops.get(r["hops"], 0) + 1
            lengths.append(len(r["text"]))
        # The most frequent single answer string is the accuracy a model gets
        # for free by always guessing it, which is the baseline any finetuned
        # score has to beat.
        majority = max(by_answer.values()) / max(len(rows), 1) if rows else 0.0
        per_split[name] = {
            "examples": len(rows),
            "by_template": dict(sorted(by_template.items())),
            "by_attribute": dict(sorted(by_attribute.items())),
            "by_hops": {str(k): v for k, v in sorted(by_hops.items())},
            "distinct_answers": len(by_answer),
            "majority_answer_share": majority,
            "mean_chars": sum(lengths) / max(len(lengths), 1),
            "max_chars": max(lengths) if lengths else 0,
        }
    return {
        "language": cfg["language"],
        "lang_code": cfg["lang_code"],
        "seed": cfg.get("seed"),
        "answer_marker": cfg["answer_marker"],
        "templates": [{"id": t["id"], "description": t["description"]}
                      for t in cfg["templates"]],
        "attributes": [{"key": a["key"], "pool": a["pool"],
                        "range": [a["low"], a["high"]]}
                       for a in cfg["attributes"]],
        "splits": per_split,
        "leakage": leakage,
    }
