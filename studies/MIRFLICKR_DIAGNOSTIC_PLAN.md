# MIRFLICKR diagnostic plan — repositioning the orthogonality contribution

## 1. What this is answering

Both reviewers (and your own checks) converge on the same weak point: the paper
claims query orthogonality "prevents attention collapse," but that claim was only
ever supported by downstream mAP, and your own Gram-matrix / attention checks show
it's false as stated — queries become perfectly orthogonal, attention still
concentrates on LL.

Reading the actual fusion head code explains **why**, mechanically:

- `CrossAttentionBottleneckHeadAdvanced` (`main/models/multi_dino_attention.py:763`,
  selected by `fusion_config.type: cross_attention_advanced`, the head used for the
  paper's real results) computes its orthogonality loss with
  `compute_ortho_loss()` directly on `self.query_tokens` — the raw learnable
  parameter, L2-normalized, Gram matrix vs identity. This is a **parameter-space**
  constraint on the queries' *identity*. It says nothing about what the queries
  actually attend to on a given image.
- The attention distribution actually used at inference — `attn_weights` from
  `self.attn(query=q, key=kv, value=kv)` — is a completely different object,
  computed from Q·Kᵀ softmax, and is never touched by that loss.
- There *is* another head already in the codebase, `CrossAttentionBottleneckHead`
  (non-"Advanced", `fusion_config.type: cross_attention_bottleneck`), whose
  orthogonality loss is instead computed on `attn_weights.mean(dim=0)` — an
  **attention-space** constraint. This one could plausibly move attention mass off
  LL. It was apparently never the one used for the headline results.

So "perfectly orthogonal queries + unchanged attention collapse" isn't a
contradiction or a bug — it's the expected outcome of regularizing the wrong
mathematical object relative to the claim being made. That's a much stronger,
more defensible story for the revision than "we tried to fix attention collapse
and partially failed": it's "we identify that query-space and attention-space
orthogonality are different constraints, show the paper used the former, and
measure what each actually does."

## 2. The six MIRFLICKR studies

All new study YAMLs live in `studies/`, follow the exact schema of the existing
VOC studies, and are pinned to the same real hyperparameters as
`mflickr_lph_vs_ortho_multiseed.yaml` (`ortho_weight=0.1`, `num_queries=4`,
`quant_weight=0.1`, `nbits=64`, `num_classes=38`, `data_dir=/content/data/mirflickr`).
All six now use `transform: basic_swt` (not `voc_swt` — `voc_swt`'s ColorJitter
was found to destabilize training) and `voc_raw4` for the raw-copies control
arm. Both are dataset-agnostic despite the name (just Resize/Crop/(SWT or
raw-copy), no ColorJitter, no VOC-specific step) and use identical
Resize/RandomResizedCrop(default scale)/RandomHorizontalFlip steps, so they
remain a valid parameter-matched pair. A third transform, `basic.yaml` (new),
is for single-spatial-domain architectures (e.g. `DINOHashBaseline` /
`config/model/dino_hashing.yaml`, one plain image in, no band stacking) — not
used by any of the six studies below yet, since none of them run that
architecture, but available if a plain single-DINO baseline study gets added
later.

| study yaml | question | reviewer | jobs | new? |
|---|---|---|---|---|
| `mflickr_lph_vs_ortho_multiseed.yaml` | Is the ortho_weight 0→0.1 mAP gain statistically robust on MIRFLICKR? | R2 (stats rigor) | 6 (2×3 seeds) | already existed |
| `mflickr_wavelet_vs_raw4_control.yaml` | Does the gain come from wavelet decomposition, or just from 4 parallel branches (capacity)? | R3 (param-matched control) | 1 | new |
| `mflickr_ortho_formulation_comparison.yaml` | Does attention-space ortho (`cross_attention_bottleneck`) actually reduce LL dominance where param-space ortho (`cross_attention_advanced`) doesn't? | emerged from investigation | 2 | new |
| `mflickr_lambda2_ablation_multidino_attention_hashing_ortho.yaml` | Sensitivity to `ortho_weight` ∈ {0, 0.01, 0.1, 1, 10} | R2 (sensitivity study) | 5 | new |
| `mflickr_num_queries_ablation_multidino_attention_hashing_ortho.yaml` | Sensitivity to `num_queries` ∈ {1, 2, 4, 8} | R2 (sensitivity study) | 4 | new |
| `mflickr_single_band_ablation.yaml` | Standalone mAP of each SWT band alone (`SingleBandNet`, no fusion at all) — tests whether LL is intrinsically more informative (DINOv2 pretrained on natural images) independent of any attention mechanism | emerged from investigation; also a clean architecture-justification ablation | 4 | new |

**Total: 22 training jobs**, all `max_iter=50`, on MIRFLICKR (the single-band jobs are
cheaper individually — one DINOv2 backbone instead of four). Use your
known per-job wall-clock from the VOC studies to budget this — architecture and
batch size are identical, only the dataset differs.

Two read-only diagnostic scripts run against the resulting checkpoints, no GPU
required for the first one:

- `studies/measure_query_orthogonality.py` — Gram matrix of the query tokens
  (already existed, dataset-agnostic, CPU-only, seconds per checkpoint).
- `studies/measure_attention_collapse.py` — **new**. Loads a checkpoint the same
  way `evaluate.py` does, runs a few real MIRFLICKR batches through it, and
  reports:
  - mean attention mass per query on each of LL/LH/HL/HH, plus the entropy of
    each query's distribution (log(4)=1.386 uniform, 0 = fully collapsed on one
    band);
  - inter-query redundancy: cosine similarity between queries' per-sample
    attention rows (not the batch mean) — distinguishes *real* functional
    collapse (queries behave identically) from queries that each individually
    favor LL but differ from each other in their secondary attention, which
    orthogonal query parameters could still cause even without moving the mean
    LL share;
  - mean pairwise cosine similarity between the 4 projected band embeddings
    *before* attention (read-out diversity, tests the "regularizer changes the
    values, not the weights" hypothesis — note this one is architecturally
    independent of the queries directly, only coupled through joint training
    dynamics, so a null result here doesn't rule out an indirect effect).

## 3. Recommended order (cheapest / most decision-relevant first)

1. **Run `mflickr_lph_vs_ortho_multiseed` first, alone.** If the gain isn't
   statistically robust on MIRFLICKR (R2's core worry), that changes what's worth
   doing next — no point building a mechanistic story around a possibly-noisy
   number. `aggregate_results.py` gives you mean ± std directly.
2. **Immediately run both diagnostic scripts** against one seed's `ortho_weight=0`
   and `=0.1` checkpoints from step 1. Free/cheap (no new training), and gives you
   the attention-collapse + read-out-diversity numbers on MIRFLICKR specifically
   (previously only checked on VOC).
3. **`mflickr_single_band_ablation`** (4 jobs, cheap — one backbone each) — run
   this in parallel with/right after step 1. Directly tests the "LL is
   intrinsically more informative" hypothesis, independent of the fusion head
   entirely. If LL-alone clearly outperforms the other three bands alone, that's
   clean, direct justification for why attention favors it — and a good
   architecture-justification ablation R2 explicitly wants regardless.
4. **`mflickr_wavelet_vs_raw4_control`** (1 job) — cheapest remaining, directly
   answers R3's capacity-vs-wavelet concern.
5. **`mflickr_ortho_formulation_comparison`** (2 jobs) — run the diagnostic
   scripts against both checkpoints too. Read this one together with step 3: if
   LL is genuinely more informative, forcing attention-space orthogonality
   (`cross_attention_bottleneck`) should *hurt* mAP (fighting a legitimate
   content-based signal), which would itself be evidence for keeping the
   query-space formulation — with a real justification instead of a preference.
6. **`mflickr_lambda2_ablation`** (5 jobs) and **`mflickr_num_queries_ablation`**
   (4 jobs) — larger, but this is literally what R2 asked for to accept after
   revision.

## 4. Colab / GCP commands

```bash
# setup (once per session/VM)
cd "Retrieval Framework"
pip install -e .
# ensure MIRFLICKR is mounted/extracted at /content/data/mirflickr with
# images/, train.txt, test.txt, database.txt (same layout the dataset class expects)

# sanity check before spending compute
python studies/run_plan.py studies/mflickr_lph_vs_ortho_multiseed.yaml --dry-run

# launch (repeat per study, in the order above)
python studies/run_plan.py studies/mflickr_lph_vs_ortho_multiseed.yaml

# aggregate mAP + bit balance, mean +/- std over seeds
python studies/aggregate_results.py studies/mflickr_lph_vs_ortho_multiseed.yaml \
    --metrics bit_balance worst_bit_balance --csv results_ortho_multiseed.csv

# Gram matrix (query-space orthogonality) -- CPU only
python studies/measure_query_orthogonality.py \
    experiments_runs/mflickr_lph_vs_ortho_multiseed_*ortho_weight=0.0*seed=333*/weights/rolling.ckpt \
    experiments_runs/mflickr_lph_vs_ortho_multiseed_*ortho_weight=0.1*seed=333*/weights/rolling.ckpt

# attention collapse + read-out diversity (needs GPU + MIRFLICKR data mounted)
python studies/measure_attention_collapse.py \
    --ckpt no_ortho=experiments_runs/mflickr_lph_vs_ortho_multiseed_*ortho_weight=0.0*seed=333*/weights/rolling.ckpt \
    --ckpt ortho=experiments_runs/mflickr_lph_vs_ortho_multiseed_*ortho_weight=0.1*seed=333*/weights/rolling.ckpt \
    --set test --n-batches 5 --bs 64
```

Repeat the last two commands' pattern for the `mflickr_ortho_formulation_comparison`
checkpoints once that study finishes, swapping in the `cross_attention_advanced`
vs `cross_attention_bottleneck` runs.

## 5. Reading the results — how they change the paper's narrative

- **If step 1's gain isn't robust (CI overlaps 0):** the "orthogonality helps
  retrieval" claim itself needs to be dropped or heavily hedged. Everything below
  becomes secondary.
- **If the gain is robust, but attention-collapse diagnostic still shows LL
  dominance for both ortho_weight=0 and 0.1:** report honestly that orthogonality
  does not resolve attention collapse, explain why mechanically (query-space vs
  attention-space, section 1), and pivot the claimed mechanism to whatever the
  read-out-diversity number (cosine similarity between bands) actually shows.
- **If `mflickr_ortho_formulation_comparison` shows `cross_attention_bottleneck`
  measurably reduces LL dominance where `cross_attention_advanced` doesn't:** this
  is your strongest new angle — a controlled comparison of two orthogonality
  formulations, with a mechanistic explanation for why only one of them can
  possibly affect the attention distribution. Worth leading the revised section
  with this, framed as a clarification of a common design ambiguity in
  cross-attention fusion, not just an ablation entry.
- **If `wavelet_vs_raw4_control` shows the raw-4-copies arm recovers most of the
  gap to full MBW-DINO:** the wavelet decomposition's real contribution is
  smaller than claimed and the paper needs to say so explicitly (this is exactly
  what R3 is testing for).
- **If `mflickr_single_band_ablation` shows LL alone clearly beats LH/HL/HH
  alone:** you have direct, mechanism-independent evidence that LL is the more
  informative band — the honest explanation for attention concentrating there
  becomes "the model is behaving rationally given unequal band informativeness,"
  not "the regularizer failed." Combine with the entropy/inter-query-redundancy
  numbers from `measure_attention_collapse.py`: if entropy is low but
  inter-query cosine similarity is also low, queries aren't functionally
  redundant even though they agree on favoring LL — worth stating explicitly
  rather than leaving "attention collapse" as a single undifferentiated verdict.
  If `mflickr_ortho_formulation_comparison`'s attention-space variant also comes
  out lower on mAP than the query-space variant, that's convergent evidence for
  keeping the current design, for the informativeness reason rather than the
  original (unsupported) "resolves collapse" reason.

## 6. GPU cost: eval policy + decoupling eval onto a cheaper GPU

All six studies above now use:

```yaml
experience.train_eval_freq: -1   # was 5
experience.test_eval_freq: 50    # was 5 -- = max_iter, so it fires once, at the final epoch
experience.fast_eval_freq: 5     # new -- cheap subset eval for a monitoring curve
experience.fast_eval_size: 500
```

Why this is safe, not just cheap:

- `train_dataset` eval (`train_eval_freq`) re-encodes the *entire training set*
  through the model every time it fires, but `principal_metric`/`eval_split` are
  hardcoded to the `test` split (`config/experience/default.yaml:23-24`) —
  train-set mAP never affects `best_score`, `best_model`, or the number
  `aggregate_results.py` reports. It's TensorBoard-only monitoring. Turning it
  off changes nothing about what gets reported.
- `config/optimizer/basic.yaml` has an empty `scheduler_on_val` — the LR
  schedule is a plain `CosineAnnealingLR` stepped every epoch regardless of
  eval, so reducing eval frequency cannot silently change training dynamics.
- The real trade-off is on `test_eval_freq`: less frequent eval means
  `best_score` (used by `aggregate_results.py`) reflects the *final* epoch
  rather than the best epoch seen during training. Evaluating only once, at
  `epoch=max_iter`, means every reported number is a final-epoch number. This
  is a **more conservative reporting convention** (no implicit best-epoch
  selection against the test set), not just a cost cut — worth stating
  explicitly in the paper's experimental setup section regardless of the
  compute angle.
- `fast_eval_freq`/`fast_eval_size` (`main/engine/batch_map.py`'s
  `build_fast_eval_subset`, wired into `main/engine/train.py`) already existed
  in the codebase but was never turned on (default `-1`). It evaluates a random
  500-image subset of the *training* set only — no full top_k=19581 gallery
  pass — so you still get a `Fast/Evaluation/*` TensorBoard curve every 5
  epochs to catch divergence/collapse early, at a fraction of the cost of a
  real test-set eval.

**Running eval on a separate, cheaper GPU later:** `evaluate.py` (repo root) is
already a fully standalone entry point — it reconstructs the model and dataset
from a checkpoint's own saved `config`, needs no optimizer/scaler/scheduler
state, and requires only a forward pass (no gradients), so it's a much lighter
GPU/memory footprint than training. With `save_model: 5` still checkpointing
every 5 epochs, nothing stops you from training on the expensive GPU with eval
turned down as above, then batch-evaluating whichever saved `epoch_*.ckpt`
files you actually want on a separate cheap/spot instance afterward:

```bash
# on the training GPU: nothing extra needed, save_model already keeps epoch_*.ckpt

# later, on a cheaper GPU (e.g. a T4/L4 spot instance), batch-evaluate:
ls experiments_runs/*/weights/epoch_50.ckpt > to_eval.txt
python evaluate.py --config to_eval.txt --parse-file \
    --set test --bs 256 --k 19581 --distance-metric hamming \
    --metric-dir results_final.txt
```

`--bs` can go higher than the training batch size here since there's no
gradient memory to share with; increasing `eval_bs` (already 200 in these
studies) similarly speeds up any in-training eval that does still run.

**Bug fixed (2026-08-09):** turning on `fast_eval_freq` surfaced a real bug in
`main/engine/batch_map.py`'s `build_fast_eval_subset` — it grouped images by
`dataset.labels` used as a raw dict key, but labels are multi-hot float
tensors for both VOC and MIRFLICKR (multi-label datasets), and a tensor's
default `__hash__` is identity-based, not value-based. Every image silently
became its own singleton "class," `eligible_classes` ended up empty, the fast
subset was empty, and `compute_all_embeddings` crashed with
`UnboundLocalError: all_q` on the empty dataloader. Fixed to group by
`dataset.instance_dict` (already built correctly per tag by every dataset
class) instead, with deduplication since one image can carry multiple tags.
Also added a clear error instead of the opaque `UnboundLocalError` if any
split's dataloader is ever empty. Verified against a standalone repro
(`/tmp/verify_fast_eval_fix.py`) since torch isn't available in this sandbox
to run the real modules directly.

### 6.1 Decision (2026-08-10): fast_eval is NOT used for epoch selection

Correlation analysis on the healthy pilot rerun (log 21, seed=333) plus the
collapsed pilot run (log 18) settled the open question from section 6.1 below:

- `fast_maphashing` vs `test_maphashing`: corr = 0.44, and `fast_maphashing`'s
  variance is ~6x smaller than `test_maphashing`'s (std 0.003 vs 0.019) — it
  barely moves regardless of what the real signal does.
- `fast_bitbalance` vs `test_bitbalance`: corr = 0.06 — no relationship.
- Decisive case: in the collapsed run (log 18), `test_bitbalance`/`test_maphashing`
  were fully frozen (0.0 / 0.7736, identical at epoch 5 and 10) while
  `fast_maphashing` stayed ~0.95 throughout — fast_eval did not detect the
  collapse at all. Likely cause: `build_fast_eval_subset`'s `min_per_class>=2`
  sampling makes the self-retrieval task structurally easy regardless of true
  embedding quality.

**Consequence:** fast_eval stays on (`fast_eval_freq: 5`) purely as a
divergence/NaN canary — cheap, still worth having — but is never used to pick
the best epoch or as a stand-in for the real test signal. Epoch selection
instead uses the offline batch-eval workflow already described above: train
with `save_model: 5` (already the default across all 7 study YAMLs),
`test_eval_freq: 50` (or `-1`, since the offline pass supersedes it) on the
expensive training GPU, then run `evaluate.py` against every saved
`epoch_*.ckpt` on a separate, cheaper GPU/session afterward to get a real
per-epoch test mAP curve and pick the actual best epoch from that — not from
`fast_eval`, and not by assuming the final epoch is best.

### 6.2 Pilot run: validate the policy before trusting it on all 6 studies

`studies/mflickr_pilot_eval_tracking.yaml` — single job (ortho_weight=0.1,
seed=333, the real config), with `test_eval_freq=5` (full eval every 5 epochs,
unlike the other 6 studies) and `fast_eval_freq=5` at the same frequency, so
the two can be compared point-for-point. Run this **first**, before the other
6 studies, to answer both open questions from this conversation in one shot:

```bash
python studies/run_plan.py studies/mflickr_pilot_eval_tracking.yaml
# after it finishes:
python studies/measure_eval_tracking.py experiments_runs/mflickr_pilot_eval_tracking_experience.seed=333/
```

`measure_eval_tracking.py` reports, from the real TensorBoard logs:

1. Whether the final epoch (50) is at or near the best test mAP observed
   during training, or trails it by some margin — this either confirms the
   "final-epoch-only" policy used by the other 6 studies, or tells you to add
   a safety-net eval point (e.g. epoch 25) instead.
2. The Pearson correlation between `fast_eval` (cheap, 500-image train-subset)
   and the real `test_eval` trajectory — high correlation means `fast_eval`
   can be trusted as a monitoring signal going forward; low/moderate
   correlation means treat it only as a divergence canary, not a quality
   proxy.

If the pilot shows the final epoch trailing the best epoch by a non-trivial
margin, revisit the `test_eval_freq: 50` choice in the other 6 study YAMLs
(e.g. switch to `test_eval_freq: 25` for two eval points) before launching
them at scale.
