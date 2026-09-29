"""Plotting only.

Every plot needs a title, both axis labels and a legend. _finish() applies all
four in one place so none of them can be forgotten.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _finish(ax, title: str, xlabel: str, ylabel: str, path: str, legend: bool = True):
    """Apply the required labelling, then save and close the figure."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fertility_vs_vocab(results: dict, lang: str, path: str):
    """Fertility against vocabulary size, with the Phase 2 parameter cost.

    `results` maps algorithm -> [(vocab_size, chars_per_token, phase2_total)].
    Both curves share the x-axis because together they are the argument for the
    chosen vocabulary: compression flattens while parameter cost keeps climbing.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, rows in sorted(results.items()):
        ax.plot([r[0] for r in rows], [r[1] for r in rows],
                marker="o", label=f"{algo} (chars/token)")
    any_algo = sorted(results)[0]
    ax2 = ax.twinx()
    ax2.plot([r[0] for r in results[any_algo]],
             [r[2] / 1e6 for r in results[any_algo]],
             linestyle="--", color="grey", label="Phase-2 params (M)")
    ax2.set_ylabel("Estimated Phase-2 parameters (millions)")
    ax2.legend(loc="lower right")
    _finish(ax, f"Tokenizer fertility vs vocabulary size - {lang}",
            "Vocabulary size", "Characters per token", path)


def plot_attrition(rows: list, lang: str, path: str):
    """Documents surviving each cleaning stage."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["stage"] for r in rows]
    docs = [r["docs"] / 1e6 for r in rows]
    ax.bar(names, docs, label="documents remaining")
    for i, v in enumerate(docs):
        ax.text(i, v, f"{v:.2f}M", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    _finish(ax, f"Documents surviving each cleaning stage - {lang}",
            "Pipeline stage", "Documents remaining (millions)", path)


def plot_zipf(ranks: list, freqs: list, lang: str, path: str):
    """Token frequency distribution on log-log axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, freqs, label="token frequency")
    _finish(ax, f"Token frequency distribution (Zipf) - {lang}",
            "Rank (log scale)", "Frequency (log scale)", path)


def plot_doc_lengths(lengths: list, lang: str, path: str):
    """Document length histogram, log-scaled on the count axis."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(lengths, bins=60, log=True, label="documents")
    _finish(ax, f"Document length distribution - {lang}",
            "Characters per document", "Number of documents (log scale)", path)


def plot_manual_split(stats: dict, lang: str, path: str):
    """Manual versus downloaded tokens, with the 20% requirement marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["manual", "downloaded"],
           [stats["manual_tokens"] / 1e6, stats["downloaded_tokens"] / 1e6],
           label="tokens")
    ax.axhline(stats["total_tokens"] * 0.20 / 1e6, linestyle="--", color="red",
               label="20% requirement")
    _finish(ax, f"Manual vs downloaded tokens - {lang}",
            "Source class", "Tokens (millions)", path)


def plot_source_mix(by_source: dict, lang: str, path: str):
    """Share of the final corpus contributed by each source."""
    items = sorted(by_source.items(), key=lambda kv: -kv[1]["bytes"])
    total = sum(v["bytes"] for _, v in items) or 1
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([k for k, _ in items],
           [v["bytes"] / total * 100 for _, v in items], label="share of bytes")
    plt.xticks(rotation=30, ha="right")
    _finish(ax, f"Corpus composition by source - {lang}",
            "Source", "Share of final corpus (%)", path)


# --------------------------------------------------------------------------
# Phase 2: training curves and attention
# --------------------------------------------------------------------------

def plot_loss_curves(history: list, lang: str, path: str):
    """Training and validation loss against step, with perplexity alongside.

    The two axes share a single legend. Letting each axis draw its own puts two
    boxes in the same corner, where they overlap and hide entries.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [h["step"] for h in history]
    l1, = ax.plot(steps, [h["train_loss"] for h in history],
                  label="training loss")
    l2, = ax.plot(steps, [h["val_loss"] for h in history],
                  label="validation loss")
    ax2 = ax.twinx()
    l3, = ax2.plot(steps, [h["val_ppl"] for h in history], linestyle="--",
                   color="grey", label="validation perplexity (right axis)")
    ax2.set_ylabel("Validation perplexity")
    ax.legend(handles=[l1, l2, l3], loc="upper right")
    ax.set_title(f"Training and validation loss - {lang}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats)")
    ax.grid(alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_lr_schedule(history: list, lang: str, path: str):
    """The learning-rate schedule actually followed."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["step"] for h in history], [h["lr"] for h in history],
            label="learning rate")
    _finish(ax, f"Learning-rate schedule - {lang}",
            "Training step", "Learning rate", path)


#: Fonts that cover Telugu and Devanagari. Matplotlib's default has neither, so
#: without one of these the tick labels render as empty boxes and the heatmap
#: cannot be read against the sentence.
INDIC_FONT_CANDIDATES = [
    "Noto Sans Telugu", "Noto Serif Telugu", "Lohit Telugu", "Gautami",
    "Noto Sans Devanagari", "Noto Serif Devanagari", "Lohit Devanagari",
    "Mangal", "Kokila", "Nirmala UI", "Arial Unicode MS",
]


def indic_font(lang: str = None):
    """An installed font that can draw Telugu or Devanagari, or None.

    Returns a FontProperties. When nothing suitable is installed the caller
    should fall back to position indices rather than emit boxes.
    """
    from matplotlib import font_manager

    # Callers pass a run name, which is the language code possibly followed by
    # a variant suffix, so match on the prefix rather than the whole string.
    want = INDIC_FONT_CANDIDATES
    if lang and lang.startswith("tel"):
        want = [f for f in want if "Telugu" in f or "Unicode" in f] + want
    elif lang and lang.startswith("nep"):
        want = [f for f in want if "Devanagari" in f or "Unicode" in f] + want
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in want:
        if name in installed:
            # Fall back to DejaVu for glyphs the Indic font lacks: Latin
            # letters and SentencePiece's word-boundary marker U+2581.
            return font_manager.FontProperties(family=[name, "DejaVu Sans"])
    return None


def plot_attention_heatmap(weights, lang: str, layer: int, heads: list,
                           tokens: list, path: str):
    """Query-vs-key attention for several heads of one layer.

    `weights` is (n_heads, T, T) for a single sequence. Tokens label the axes so
    the picture can be read against the sentence rather than against indices;
    where no Indic-capable font is installed the labels fall back to positions,
    which is uninformative but at least legible.
    """
    font = indic_font(lang)
    n = len(heads)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    T = min(len(tokens), weights.shape[-1])
    for col, head in enumerate(heads):
        ax = axes[0][col]
        im = ax.imshow(weights[head][:T, :T], aspect="auto", origin="upper",
                       cmap="viridis", vmin=0.0)
        ax.set_title(f"layer {layer}, head {head}")
        ax.set_xlabel("Key position")
        if col == 0:
            ax.set_ylabel("Query position")
        step = max(1, T // 12)
        ax.set_xticks(range(0, T, step))
        ax.set_yticks(range(0, T, step))
        labels = ([tokens[i] for i in range(0, T, step)] if font
                  else [str(i) for i in range(0, T, step)])
        ax.set_xticklabels(labels, rotation=90, fontsize=6, fontproperties=font)
        ax.set_yticklabels(labels, fontsize=6, fontproperties=font)
        fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    fig.suptitle(f"Attention weights - {lang}, layer {layer}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_attention_summary(rows: list, lang: str, path: str, metric: str,
                           ylabel: str):
    """Per-head entropy or mean distance, grouped by layer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = sorted({r["layer"] for r in rows})
    for head in sorted({r["head"] for r in rows}):
        ys = [next(r[metric] for r in rows
                   if r["layer"] == l and r["head"] == head) for l in layers]
        ax.plot(layers, ys, marker="o", label=f"head {head}")
    _finish(ax, f"Attention {ylabel.lower()} by layer - {lang}",
            "Layer", ylabel, path)


def plot_reasoning_accuracy(by_template: dict, lang: str, path: str,
                            baseline: float = None):
    """Pretrained against finetuned accuracy, one group of bars per family.

    `by_template` maps a family id to {"pretrained": acc, "finetuned": acc}.
    The chance line is drawn where a baseline is given, because a bar sitting on
    it means the model is guessing, not reasoning.
    """
    families = list(by_template)
    x = range(len(families))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(families)), 4.2))
    ax.bar([i - width / 2 for i in x],
           [by_template[f]["pretrained"] for f in families], width,
           label="pretrained")
    ax.bar([i + width / 2 for i in x],
           [by_template[f]["finetuned"] for f in families], width,
           label="finetuned")
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", linewidth=1, color="grey",
                   label=f"chance ({baseline:.0%})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    _finish(ax, f"Reasoning accuracy by template family - {lang}",
            "Template family", "Exact-match accuracy", path)


def plot_attention_shift(pre_layers: list, ft_layers: list, lang: str,
                         path: str, metric: str, ylabel: str):
    """Per-layer attention statistic before and after finetuning.

    Both curves on one axis: what matters is whether finetuning moved attention
    and in which direction, which a difference plot alone would hide.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["layer"] for r in pre_layers], [r[metric] for r in pre_layers],
            marker="o", label="pretrained")
    ax.plot([r["layer"] for r in ft_layers], [r[metric] for r in ft_layers],
            marker="s", label="finetuned")
    ax.set_xticks([r["layer"] for r in pre_layers])
    _finish(ax, f"{ylabel} by layer, before and after finetuning - {lang}",
            "Layer", ylabel, path)


def plot_metric_comparison(labels: list, series: dict, title: str,
                           ylabel: str, path: str):
    """Grouped bars for a handful of named runs over the same metrics.

    Used for the ablation, where the standard and no-position models are
    compared on several generation metrics at once.
    """
    n = len(series)
    width = 0.8 / max(n, 1)
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(labels)), 4.2))
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * width
        ax.bar([j + offset for j in x], values, width, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _finish(ax, title, "", ylabel, path)
