# Monolingual language models — Telugu and Nepali

Two independent decoder-only transformers and the data pipelines behind them.
Phase 1 built the corpora and tokenizers; Phase 2 implements the architecture
from primitive PyTorch layers, pretrains both models and evaluates them; Phase 3
finetunes each one on a synthetic comparative-reasoning set in its own language
and compares what the two resource tiers can learn.

Telugu is Model H. It has plenty of public text and, usefully, a Unicode block of
its own, so deciding whether a document is Telugu is close to a solved problem.

Nepali is Model L. It shares Devanagari with Hindi, Marathi and Maithili, so the
same question is genuinely hard, and answering it took an ensemble of script
ratio, GlotLID and a function-word discriminator. The gap between the two
languages, in difficulty and in how much data exists at all, is the most
interesting thing this phase turned up.

The two corpora share no documents, no vocabulary and no tokenizer. Every stage
runs per language, from its own config, into its own directory.

| | Telugu (H) | Nepali (L) |
|---|---:|---:|
| Training tokens | **639,877,839** | **497,000,869** |
| Documents | 1,097,873 | ~1.2M |
| Manual share | 15.26% | 8.34% |
| Tokenizer | unigram, 8k | unigram, 32k |
| Bytes per token | 10.00 | 13.01 |

**Submitted report: [`report/final_report.pdf`](report/final_report.pdf)** — the
consolidated 10-page write-up covering all three phases and the bonus ablation.

Longer per-phase working notes, kept for reference:
[`report/phase1_report.md`](report/phase1_report.md),
[`report/phase2_report.md`](report/phase2_report.md),
[`report/final_report.md`](report/final_report.md).

## The models

Two decoder-only transformers, ~25M parameters each, written from primitive
PyTorch layers. No `nn.Transformer`, no `nn.MultiheadAttention`, no
`F.scaled_dot_product_attention` — the projections, head reshape, scaled dot
product, causal mask and output projection are all explicit in
`common/model/attention.py`.

| | Telugu (H) | Nepali (L) |
|---|---:|---:|
| Parameters | 24,562,944 | 24,906,624 |
| Layers × heads | 12 × 6 | 7 × 6 |
| `d_model` / context | 384 / 512 | 384 / 512 |
| Embedding share | 13.3% | 50.1% |
| Trained on | 1,356.6M tokens (2.12 epochs) | 1,048.6M tokens (2.11 epochs) |
| Test perplexity | 24.15 | 59.37 |
| **Test bits per byte** | **0.4588** | **0.4529** |

Width and head count are identical; only depth differs, absorbing the difference
between an 8,000-piece and a 32,000-piece vocabulary so both land at ~25M.

Perplexity is not comparable between the two — their tokenizers differ, so a
token means something different in each. Bits-per-byte normalises by the
underlying text, and on it the two models are within 1.3% of each other despite
Model L having 22% less data and five fewer layers.

## Reasoning

Each model is finetuned on its own synthetic comparative-reasoning set, written
in its own language and script. Nothing is downloaded and nothing is translated:
`scripts/30_make_reasoning.py` draws the numbers or the ordering first and
renders the sentences from it, so every gold answer is known by construction.

| | per language |
|---|---|
| Train / validation / test | 22,000 / 1,500 / 3,600 |
| Template families | 5 |
| Statements to combine | 1 to 3 |
| Leakage controls | disjoint entity pools per split, plus one relation pattern held out of training entirely |

The families run from a single comparison of two stated values up to a chain of
three inequalities over four entities with no numbers given at all. Statements
are shuffled inside every example, so the answer cannot be read off the surface
order — naming whichever entity appears first scores 26.9% for Telugu and 34.5%
for Nepali, both below the 38.4% chance rate.

| | Telugu (H) | Nepali (L) |
|---|---:|---:|
| Pretrained exact match | 0.00% | 0.00% |
| **Finetuned exact match** | **40.00%** | **9.11%** |
| Finetuned choice accuracy | 41.75% | 35.11% |
| Random-choice baseline | 38.43% | 38.43% |

Read against that baseline the result is largely negative: finetuning taught
both models the output format but, apart from one family for Telugu
(`transitive_3`, +7.4 points over chance), not the comparison itself.
[`report/final_report.md`](report/final_report.md) takes this apart against three
baselines and explains why Model L does worse.

Every string a generated example can contain lives in
`<language>/configs/reasoning.yaml` rather than in the generator, so the
phrasing can be reviewed and corrected without touching code. The generator is
seeded, so the dataset reproduces byte for byte.

## Data

The corpora are far too large for git, so they live on Drive.

**[Corpora, tokenized arrays and dataset card](https://drive.google.com/drive/folders/1AJO92VfuIbdJmp5GSaVmJz8BzUPa31I_)**

| | path |
|---|---|
| Telugu corpus | `telugu/5_final/` |
| Telugu tokenized | `telugu/6_tokenized/` |
| Telugu pretrained | [`tel_best.pt`](https://drive.google.com/file/d/1d6-hMx5m91LojrqwT1YYgCW4uuVxcWL_/view?usp=sharing) (295 MB) |
| Telugu finetuned | [`tel_ft_best.pt`](https://drive.google.com/file/d/1T1cS79d2VnK29_xLYMo2Hui2KLYE7gOx/view?usp=sharing) |
| Telugu ablated (no positions) | [`tel_nopos_best.pt`](https://drive.google.com/file/d/18BPIhTI-GnmYrpQ0pp0ZpdTvhxY9Nzbo/view?usp=sharing) |
| Nepali corpus | `nepali/5_final/` |
| Nepali tokenized | `nepali/6_tokenized/` |
| Nepali pretrained | [`nep_best.pt`](https://drive.google.com/file/d/1enzNOOGkLtPIlS1a1HQZGt5TZrJcDP4w/view?usp=sharing) (300 MB) |
| Nepali finetuned | [`nep_ft_best.pt`](https://drive.google.com/file/d/1gOvLtRstiLDbHCoHB0qfLXG4J4DnHzfS/view?usp=sharing) |
| Nepali ablated (no positions) | [`nep_nopos_best.pt`](https://drive.google.com/file/d/1N1oyV1WHniLPM-83egJVdShK3XSMfdGR/view?usp=sharing) |
| Dataset card | `DATASET_CARD.md` |

Finetuned checkpoints are written in the same format as the pretrained ones —
weights, optimizer state, step counter, history and both configs — so
`scripts/31_finetune.py` resumes from one rather than starting over.

Raw crawled HTML is not included.

Each language directory on Drive follows the same layout:

```
<language>/
  1_raw/          everything collected, before any cleaning
    downloaded/     public corpora, one folder per source
    manual/         collected by us
      web_crawl/<host>/   our own crawler, one folder per site
      ocr_books/          our own OCR of scanned books
      pdf_documents/      text-layer PDFs
  2_splits/       train/val/test, held out BEFORE cleaning
  3_clean/        after language id, normalisation and quality filters
  4_dedup/        after duplicate removal and decontamination
  5_final/        the corpus
  6_tokenized/    uint16 token arrays, offsets, exact token counts
  7_checkpoints/  pretrained and finetuned weights, one config beside each
  9_reasoning/    the generated reasoning splits and their dataset card
```

Shards are `.jsonl.zst`, one JSON document per line. Every document carries its
own provenance: source, licence, URL, fetch date, and which filters it passed.
That means the composition of the corpus can be checked from the data itself
rather than taken on trust from the folder names.

## Repository

```
common/
  collect/      crawler, HTML extraction, OCR, PDF text, HuggingFace download
  clean/        normalisation, language id, quality filters, dedup, PII
  tokenizer/    library training, evaluation, from-scratch implementations
  io/           sharded corpus format, provenance schema, checkpointing
  model/        attention, transformer block, decoder LM, config
  train/        token batching, training loop, checkpointing
  eval/         perplexity and bits-per-byte, generation, text metrics, attention
  reasoning/    synthetic dataset generation, finetuning batches, scoring
scripts/        numbered pipeline stages, run in order
telugu/         configs, tokenizer and vocabulary for Model H
nepali/         configs, tokenizer and vocabulary for Model L
report/         the Phase 1 report, with all figures and tables
```

Each language directory holds its own `configs/` (sources, cleaning thresholds,
tokenizer hyperparameters), `tokenizer/` (model file, vocabulary, sweep results)
and `splits/` (the split manifest). Nothing crosses between them.

## Pipeline

| | stage | |
|---|---|---|
| 01 | download | public corpora from HuggingFace |
| 02, 02b | crawl, extract | our own crawler, then HTML to article text |
| 03, 03b | OCR, PDF text | scanned books, and text-layer documents |
| 04 | split | val/test held out **before** cleaning |
| 05 | clean | language id, Unicode normalisation, quality filters, PII |
| 06 | dedup | exact and MinHash near-duplicate removal, decontamination |
| 07 | finalize | blocklists, source-mix capping, corpus sizing |
| 08 | tokenizers | trained and swept over algorithms and vocabulary sizes |
| 09 | tokenize | encode to uint16, exact token counts |
| 10, 11, 12 | report | figures and tables, dataset card, packaging |

### Phase 2

| | stage | |
|---|---|---|
| 19 | sanity | parameter count, `ln(V)` initial loss, causal-mask check |
| 20 | train | pretraining, resumable from checkpoints |
| 21 | evaluate | cross-entropy, perplexity, bits-per-byte, loss curves |
| 22 | generate | greedy and temperature decoding, BLEU, chrF++, ROUGE-L, diversity |
| 23 | attention | heatmaps, per-head entropy and mean attention distance |
| 24 | report | assembles `report/phase2_report.md` |

### Phase 3

| | stage | |
|---|---|---|
| 30 | reasoning data | generates the synthetic dataset, checks the leakage controls |
| 31 | finetune | continues from the pretrained checkpoint, resumable |
| 32 | evaluate | pretrained against finetuned exact match, per family and per hop |
| 33 | attention | the same toolkit on both checkpoints, over reasoning prompts |
| 34 | ablation | optional bonus: retrain without positional embeddings |
| 35 | report | assembles `report/final_report.md` |

### The bonus ablation

Both languages were retrained with `use_positional_embeddings: false` — 196,608
fewer parameters, 0.8% of the model, everything else held fixed. Compared at a
matched 6,000 steps, removing positions costs **+0.094 nats for Telugu and
+0.128 for Nepali**, far less than the permutation-invariance argument suggests,
because the causal mask already carries positional information. What does break
is attention structure: mean entropy rises 1.85 → 2.22 (Telugu), the sharp
local-processing band disappears, long-range reach falls from 16.0 to 12.1
positions, and greedy 4-gram repetition rises by a third.

Both models are ~25M parameters with `d_model` 384 and 6 heads. Only depth
differs — 12 layers for Telugu, 7 for Nepali — because Nepali's 32k vocabulary
costs four times as much in the embedding, and holding width and head count
fixed leaves one architectural variable between the two rather than three.

Attention is written out in `common/model/attention.py`: projections, head
reshape, scaled dot product, causal mask and output projection, with no
`nn.MultiheadAttention` and no `F.scaled_dot_product_attention`.

Two decisions are worth calling out. Validation and test are carved out at stage
04, before any cleaning touches the data, so no cleaning threshold can leak
across the split boundary; stage 06 then drops training documents that duplicate
held-out text. And deduplication runs across all sources at once rather than one
source at a time, which is how a page our crawler fetched gets matched against
the copy of it sitting inside a downloaded corpus. Where the two collide, the
copy we collected is the one kept.

## Running it

```bash
pip install -r requirements.txt     # or: bash setup_colab.sh on Colab
python scripts/run_all.py           # Phase 1: collection through tokenizers
python scripts/run_phase2.py        # Phase 2: sanity, training, evaluation
python scripts/run_phase3.py        # Phase 3: reasoning data, finetuning, analysis
```

Phase 3 needs the Phase 1 tokenizers and the Phase 2 checkpoints in the work
directory; everything else it generates itself. The bonus ablation is a full
retrain and is therefore opt-in:

```bash
python scripts/run_phase3.py --ablation      # adds stage 34 on Telugu
python scripts/34_ablation.py --lang tel --skip-train    # compare only
```

On Kaggle, where the checkpoints have to come down from Drive first, one script
does the whole thing — packages, downloads, the tokenizer-against-checkpoint
check, all four stages for both languages, and a tarball of the results:

```bash
python scripts/kaggle_phase3.py \
    --tel-checkpoint <drive link> \
    --nep-checkpoint <drive link> \
    --tel-tokenizer  <drive link>
```

Downloads are skipped when the file is already in place and training resumes
from its last checkpoint, so re-running after a session dies costs nothing.

Before committing GPU time, the checks that catch the expensive mistakes:

```bash
python scripts/19_sanity.py --lang tel
```

That verifies the parameter count against the config, that the loss before
training equals `ln(vocab_size)`, and that altering a future token leaves
earlier logits bit-identical.

Single stages, and a progress report that is safe to run while a crawl is going:

```bash
python scripts/run_all.py --only 06,07 --lang tel
python scripts/status.py
```

Every stage checkpoints. That mattered more than expected, since most of this ran
on Colab and sessions died repeatedly; an interrupted run resumes instead of
starting over. Stages 04 to 07 clear their own output before rewriting it, so
re-running one is safe and gives the same counts every time.
