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
All the fusion-based studies use `transform: basic_swt` (not `voc_swt` —
`voc_swt`'s ColorJitter was found to destabilize training).

**R3's capacity-matched control (2026-08-11): raw*4 dropped, replaced by a
single ViT-B backbone.** `mflickr_wavelet_vs_raw4_control.yaml` (4 identical
raw copies of the image through the same 4-branch architecture) was the
original design, but the advisors rejected image duplication as a control
design on principle -- the file is kept for reference but is no longer part
of the plan. `mflickr_vitb_capacity_control.yaml` (new) replaces it: a single
`DINOHashBaseline` (`config/model/dino_hashing.yaml`) run with
`dino_backbone: dinov2_vitb14` (~86M params, vs ~21M x 4 = ~84M for the four
unshared ViT-S branches in MBW-DINO) -- no duplication, no fusion head, one
coherent backbone whose capacity is genuinely exercised end to end. Uses the
new `transform: basic` (single spatial domain: Resize/RandomCrop/
RandomHorizontalFlip, no ColorJitter, no aggressive RandomResizedCrop
scale/ratio, no band stacking -- built for exactly this case).

| study yaml | question | reviewer | jobs | new? |
|---|---|---|---|---|
| `mflickr_lph_vs_ortho_multiseed.yaml` | Is the ortho_weight 0→0.1 mAP gain statistically robust on MIRFLICKR? | R2 (stats rigor) | 6 (2×3 seeds) | already existed |
| `mflickr_vitb_capacity_control.yaml` | Does the full multi-branch + fusion architecture beat a single, roughly parameter-matched ViT-B backbone with no wavelet decomposition, no fusion, no ortho? | R3 (param-matched control) | 1 | new (replaces `mflickr_wavelet_vs_raw4_control.yaml`, rejected by advisors) |
| `mflickr_ortho_formulation_comparison.yaml` | Does attention-space ortho (`cross_attention_bottleneck`) actually reduce LL dominance where param-space ortho (`cross_attention_advanced`) doesn't? | emerged from investigation | 1 (was 2 -- see dedup note below) | new |
| `mflickr_lambda2_ablation_multidino_attention_hashing_ortho.yaml` | Sensitivity to `ortho_weight` ∈ {0, 0.01, 0.1, 1, 10} | R2 (sensitivity study) | 3 (was 5 -- see dedup note below) | new |
| `mflickr_num_queries_ablation_multidino_attention_hashing_ortho.yaml` | Sensitivity to `num_queries` ∈ {1, 2, 4, 8} | R2 (sensitivity study) | 3 (was 4 -- see dedup note below) | new |
| `mflickr_single_band_ablation.yaml` | Standalone mAP of each SWT band alone (`SingleBandNet`, no fusion at all) — tests whether LL is intrinsically more informative (DINOv2 pretrained on natural images) independent of any attention mechanism | emerged from investigation; also a clean architecture-justification ablation | 4 | new |

**Total: 18 training jobs** (down from a naive 22), all `max_iter=50`, on
MIRFLICKR (the single-band jobs are cheaper individually — one DINOv2 backbone
instead of four). Use your known per-job wall-clock from the VOC studies to
budget this — architecture and batch size are identical, only the dataset
differs.

**Dedup note (2026-08-11):** `mflickr_lph_vs_ortho_multiseed`'s
`ortho_weight=0.1, seed=333` job (`fusion_config.type` defaults to
`cross_attention_advanced`) is, by construction, an exact config match for one
grid point in each of `mflickr_lambda2_ablation` (`ortho_weight=0.1`),
`mflickr_num_queries_ablation` (`num_queries=4`), and
`mflickr_ortho_formulation_comparison` (`type=cross_attention_advanced`) --
and `mflickr_lambda2_ablation`'s `ortho_weight=0.0` point matches
`mflickr_lph_vs_ortho_multiseed`'s `ortho_weight=0.0, seed=333` job too. All
four sweeps have been trimmed to exclude these already-trained points; reuse
the existing `mflickr_lph_vs_ortho_multiseed` checkpoints for those data
points in any curve, table, or diagnostic-script comparison instead of
retraining. Each yaml has an inline comment with the exact checkpoint glob to
reuse.

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
4. **`mflickr_vitb_capacity_control`** (1 job) — cheapest remaining, directly
   answers R3's capacity concern with a single ViT-B backbone instead of image
   duplication.
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

## 4.1 Result: `mflickr_lph_vs_ortho_multiseed` (2026-08-10)

All 6 runs completed, `test_bitbalance` never collapsed (range ~0.16-0.70
across all epochs of all 6 runs) and best-epoch vs final-epoch mAP gap is
<=0.3pt for every run -- both the worker-seeding fix and the
final-epoch-only reporting policy are validated by this study.

Best-epoch mAP (`evaluate_all_checkpoints.py`, `--metric map_level0`), paired
by seed:

| seed | ortho=0.0 | ortho=0.1 | diff (0.1-0.0) |
|---|---|---|---|
| 111 | 0.8436 | 0.8464 | +0.0027 |
| 222 | 0.8364 | 0.8299 | -0.0064 |
| 333 | 0.8549 | 0.8636 | +0.0087 |

mean +/- std: ortho=0.0 -> 0.8450 +/- 0.0094, ortho=0.1 -> 0.8466 +/- 0.0169.
Paired diff: **+0.0017 +/- 0.0076** (best-epoch), +0.0022 +/- 0.0085
(final-epoch). Sign flips at seed 222; std is ~4.5x the mean.

**This is the R2-relevant verdict: with n=3 seeds, the ortho_weight 0->0.1
gain on MIRFLICKR is not distinguishable from noise.** Per section 5 below,
this means the "orthogonality helps retrieval" claim needs to be dropped or
heavily hedged as the paper's headline result -- the honest framing is the
mechanistic one (query-space vs attention-space orthogonality, section 1),
not a performance claim. Training stability/reproducibility is a separate,
genuinely positive result worth reporting on its own (the seeding + clip_grad
fixes made a previously non-deterministic setup reproducible), but it doesn't
rescue the mAP-gain claim.

## 4.2 Result: attention diagnostics (2026-08-11) — there is no attention collapse

Run on `mflickr_lph_vs_ortho_multiseed`'s seed=333 checkpoints (epoch 50, 320
test images), ortho_weight 0.0 vs 0.1.

**Query orthogonality (`measure_query_orthogonality.py`)** — behaves exactly as
designed: ortho=0.0 gives mean |off-diagonal| 0.0424 (max 0.0999), i.e. the
queries are *already* nearly orthogonal without any regularizer; ortho=0.1
gives 0.0000. Confirms the earlier finding that there was little query
collapse to prevent in the first place.

**Attention distribution (`measure_attention_collapse.py`)** — no collapse, at
either setting:

| | ortho=0.0 | ortho=0.1 |
|---|---|---|
| attention share LL / LH / HL / HH | 0.258 / 0.252 / 0.247 / 0.242 | 0.258 / 0.250 / 0.250 / 0.243 |
| per-query entropy | 1.386 = log(4) (max) | 1.386 = log(4) (max) |
| LL share std across the 8 heads | 0.0035 | 0.0027 |
| band read-out cosine (mean off-diag) | 0.7156 | 0.6819 |

Entropy is at its theoretical maximum and heads do not specialize. **The
premise of the paper's contribution -- that attention collapses onto LL and
needs correcting -- does not hold on MIRFLICKR.** Together with the two other
findings (no meaningful query collapse; no statistically significant mAP gain,
section 4.1), the regularizer was addressing a problem that isn't there.

**Why the original VOC figure showed total LL dominance:** that analysis
applied `scores = raw_scores * 10.0` before the softmax, while
`nn.MultiheadAttention` scales by `1/sqrt(head_dim) = 1/sqrt(48)` -- a 69x
inflation of the logits. Feeding the measured MIRFLICKR shares through that
same 69x factor reproduces the published-style figure almost exactly
(LL=0.875, LH=0.100, HL=0.019, HH=0.006, entropy 0.66 bits). Multiplying
logits before a softmax is a temperature change, not a gain adjustment: it
manufactures peakedness. The *direction* (LL mildly favored) is real; the
*magnitude* is negligible.

**Two methodology fixes made while establishing this** (both in
`measure_attention_collapse.py`):
- Per-head capture (`average_attn_weights=False`). The PyTorch default
  averages over all 8 heads before returning, which could have hidden genuine
  per-head specialization. It didn't -- heads agree -- but the check was
  necessary.
- Inter-query similarity is now computed on *mean-centered* attention rows.
  Raw cosine between near-uniform distributions is ~1.0 by construction (the
  shared 1/4 constant dominates): two queries with diametrically *opposite*
  preferences still score 0.999. The original "inter-query cosine = 1.000 =>
  functional collapse" reading was an artifact of that. Centered, the queries
  do agree (~0.995) but on a deviation of only ~0.006 -- "they agree on a
  negligible preference", not "they have collapsed onto a strong shared
  pattern".
- The script now also reports the three distinct levels that the original
  claim conflated: (1) `q_i` raw query tokens (what the ortho loss
  constrains), (2) `W_q @ q_i + b_q` projected queries (what attention
  actually uses; `b_q` is shared across queries and `compute_ortho_loss`
  normalizes `q_i`, so level 1 orthogonality does not imply level 2
  distinguishability), (3) the attention rows themselves.

**VOC re-measured with the corrected scaling (2026-08-11): same verdict, and
stronger.** `voc_multidino_64bits_v2_ortho_01`, epoch 40, `use_all_tokens=False`
(one token per band, same structural regime as MIRFLICKR): per-query entropy
1.385-1.386 (max = log(4) = 1.386), LL share 0.265, LL-share std across the 8
heads 0.0032. No attention collapse on VOC either -- the original figure was
the 69x logit inflation, confirmed.

What makes the VOC result *stronger* than MIRFLICKR's is the band read-out
diversity, which differs sharply between the two datasets:

| band read-out cosine | VOC (ortho=0.1) | MIRFLICKR (ortho=0.1) |
|---|---|---|
| LL vs the three detail bands | **0.139** | 0.489 |
| detail bands among themselves (LH/HL/HH) | 0.781 | 0.875 |
| global mean off-diagonal | 0.460 | 0.682 |

On VOC, LL carries content nearly orthogonal to the detail bands (cosine
0.12-0.18) -- there genuinely *is* something to differentiate -- and the
attention still does not differentiate it. So the finding is not "attention is
uniform because the inputs are interchangeable"; it is "attention is uniform
even when the inputs are clearly distinct". The routing mechanism is inactive
regardless of what it is given.

Secondary observation, consistent across both datasets: LH/HL/HH are strongly
redundant with each other (0.78-0.87) while LL stands apart. Four branches
look like more than the decomposition needs -- direct motivation for
`mflickr_single_band_ablation` and `mflickr_vitb_capacity_control`.

### 4.2.1 Root cause found: the queries never grow, so the softmax cannot sharpen

Full three-level measurement on the VOC ortho=0.1 checkpoint settles *why* the
attention is uniform. The earlier "the shared bias `b_q` dominates the
projected queries" hypothesis is **refuted**: measured
`||W_q @ q_i|| / ||b_q||` = 11.13, so the bias is negligible, and the queries
remain distinct after projection (level-2 mean off-diagonal cosine 0.2124,
per-head 0.18-0.26). Direction is not the problem.

**Magnitude is.** The query tokens never left their initialization scale:

| | value |
|---|---|
| norm at init (`trunc_normal_(std=0.02)`, dim 384) | 0.3919 |
| measured after 40 epochs | 0.3999, 0.3775, 0.3996, 0.4047 |
| growth ratio | **1.009** |

Three converging reasons, all visible in the code:
- `compute_ortho_loss()` applies `F.normalize(Q)` before the Gram matrix, so
  the loss is **scale-invariant** -- it constrains direction and exerts
  literally zero gradient pressure on `||q_i||`.
- `config/optimizer/basic.yaml` uses AdamW with `weight_decay: 0.0005`, which
  actively shrinks them.
- `lr: 1e-5`, further annealed by CosineAnnealingLR.

The consequence is arithmetic. Small `||q||` -> small scores (measured
`mean|score|`=0.0805, within-row spread 0.0952) -> softmax of a spread that
small is necessarily near-uniform. Predicted from the measured spread:
LL=0.268, others=0.244. Measured: LL=0.265, LH=0.247, HL=0.242, HH=0.246. The
model sits exactly where the math puts it.

For genuinely selective attention the query norms would need to be 7-26x
larger:

| LL would capture | required within-row spread | required `||q||` |
|---|---|---|
| 40% | 0.69 | ~2.8 |
| 60% | 1.50 | ~6.2 |
| 80% | 2.48 | ~10.2 |

**This is the defensible mechanistic contribution:** the orthogonality
regularizer constrains the *direction* of the query tokens, while what
actually governs attention selectivity is their *magnitude* -- a quantity
nothing in the objective constrains and that weight decay reduces. That single
fact explains, simultaneously, why the queries are perfectly orthogonal, why
the attention stays uniform, and why the mAP gain is statistically nil. It is
also a general design lesson about cross-attention bottlenecks, not a
dataset-specific quirk.

It is directly testable, and each test is a short run:
- exclude `query_tokens` from weight decay (optimizer param group);
- `F.normalize` the queries inside `forward()` and add a learned temperature
  on the scores (decouples direction from sharpness explicitly);
- initialize `query_tokens` at a larger scale and check whether attention
  becomes selective and whether that helps or hurts mAP.

**Still open:** seeds 111/222 not yet measured on MIRFLICKR (checkpoints
exist; cheap). No VOC `ortho_weight=0` counterpart measured, so the VOC
numbers above are single-arm. The three fixes above are untested.

**Bigger question this raises:** uniform attention means the bottleneck is not
routing anything, and the band read-out cosines show LH/HL/HH are largely
redundant with each other (0.93 / 0.88 / 0.82-0.89). This suggests the fusion
head contributes little -- which makes
`mflickr_vitb_capacity_control` (section 2) the most decision-relevant study
remaining, not a formality.

## 4.3 `mflickr_vitb_capacity_control` first attempt (2026-08-11): INVALID, bug found

First run came out at ~79% mAP, far below MBW-DINO's ~84.5%. That number is not
usable: `main/models/dino_baseline.py`'s `DINOHashBaseline.forward` called

```python
with torch.set_grad_enabled(not getattr(self.backbone, 'frozen', True)):
```

but nothing anywhere ever set `.frozen` on the backbone -- the constructor's
`frozen` argument only flips `requires_grad`. The `getattr` therefore always
fell through to its default `True`, so gradients were disabled on every
forward and **the backbone stayed at its pretrained weights no matter what**.
The run trained only the hash head (`Linear(768->64) + BatchNorm1d`) on frozen
features, while MBW-DINO fine-tunes all four backbones (`frozen: False`). ~79%
is exactly what a frozen ViT-B + linear head should give, so the comparison
said nothing about capacity.

Fixed by storing the flag on `self` and respecting the ambient grad mode
(`torch.is_grad_enabled() and not self.frozen`, so it cannot re-enable grad
inside `evaluate.py`'s `no_grad` block). `config/model/dino_hashing.yaml` sets
no `frozen` key, so the default `False` now genuinely fine-tunes. **Re-run the
study.**

**Check before trusting any older baseline number:** if the accepted MBW-DINO
paper's single-DINO baseline was produced with this same class, that baseline
was a frozen-backbone number and the reported margin over it was inflated by
the same mechanism. Worth verifying against the published tables.

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
- **If `mflickr_vitb_capacity_control` (single ViT-B, ~parameter-matched) comes
  close to or beats full MBW-DINO:** the whole multi-branch + fusion design's
  real contribution over raw capacity is smaller than claimed and the paper
  needs to say so explicitly (this is exactly what R3 is testing for).
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

# later, on a cheaper GPU (e.g. a T4/L4 spot instance), batch-evaluate every
# saved epoch (not just the final one) and get the real best epoch per run:
python studies/evaluate_all_checkpoints.py studies/mflickr_lph_vs_ortho_multiseed.yaml \
    --set test --bs 256 --k 19581 --distance-metric hamming \
    --csv results_lph_vs_ortho_per_epoch.csv
```

`studies/evaluate_all_checkpoints.py` (new) walks every run directory of a
study, evaluates each `weights/epoch_*.ckpt` via `evaluate.py`'s own
`load_and_evaluate()`, and reports the best epoch per run by `--metric`
(default `map_level0`) alongside the final epoch, flagging when they differ.
This is the tool that actually implements the section 6.1 decision -- it
replaces both fast_eval and the final-epoch assumption as the source of truth
for epoch selection.

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
