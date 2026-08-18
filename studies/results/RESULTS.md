# Results log — MIRFLICKR / VOC revision

Everything measured during the ACIVS26 revision, kept in one place so no number
has to be recovered from a chat log. Raw outputs live beside this file; the
interpretation and the experiment plan live in `../MIRFLICKR_DIAGNOSTIC_PLAN.md`.

| file | what it is |
|---|---|
| **`all_runs_metrics.csv`** | **one row per run: best epoch + final epoch, all four metrics, swept params parsed from the run name. Generated — never edit by hand.** |
| `consolidate_metrics.py` | regenerates `all_runs_metrics.csv` from every `*_per_epoch.csv` here. Idempotent: drop a new study CSV in and rerun. |
| `diagnostics_metrics.csv` | secondary/diagnostic scalars (attention shares, entropies, query norms, band statistics) in long format |
| `lph_vs_ortho_multiseed_per_epoch.csv` | 6 runs (ortho 0.0/0.1 × seeds 111/222/333), every saved epoch, from `evaluate_all_checkpoints.py` |
| `vitb_capacity_control_per_epoch.csv` | ViT-B capacity control, every saved epoch |
| `wavelet_type_ablation_per_epoch.csv` | 3 runs (haar/db4/bior4.4), seed 333, every saved epoch (section 5b) |
| `num_queries_sb96_per_epoch.csv` | 4 runs (N=1/2/4/8), sub_batch=96 fixed, seed 333, every saved epoch (section 5c) |
| `mflickr_final_headline_per_epoch.csv` | MIRFLICKR final headline (32/64/128 bits), num_queries=1, seed 333, no eval during training, `--k 19581` (section 5d) |
| `coco_database.txt` | COCO's `database.txt` as actually used for training — its line count (117,218) is the confirmed `--k` value for COCO's mAP@ALL pass |
| `final_headline_training_curves.csv` | per-epoch `HashLoss`/`Ortho_Loss`/`total_loss` for all 8 final-headline jobs (mflickr×3, voc×3, coco×2), parsed from the raw Colab training log via `studies/parse_training_log.py` — kept for the thesis appendix's training curves |
| `diagnostics_attention_2026-08-11.txt` | verbatim `measure_query_orthogonality.py` + `measure_attention_collapse.py` output (MIRFLICKR + VOC) |
| `swt_transform_check_2026-08-12.txt` | verbatim `verify_swt_transform.py` output |

Workflow for new results: drop the study's `*_per_epoch.csv` here, run
`python studies/results/consolidate_metrics.py`, and the summary table picks it
up. Diagnostic scalars go into `diagnostics_metrics.csv` (long format, one
metric per row, so new metrics never require changing the schema).

All mAP figures below are `maphashing_level0` (the hashing-literature mAP@topk
from `calculate_maphashing`), best epoch per run, `top_k=19581`, hamming.
`map_level0` (torchmetrics `RetrievalMAP`) is also in the CSVs and occasionally
picks a different best epoch — the two disagree on 2 of the 6 multiseed runs.

---

## 0. What's actually noisy here, and how to write around it

Two different things get called "noise" in this project and they must not be
conflated — one is real and small, the other is real and comparable in size to
the effects we're reporting.

**Within-seed reproducibility: CORRECTION (2026-08-17) — the ~0.0001 figure
below does NOT generalize to the whole project.** It was measured on N=1
(`mflickr_subbatch_vs_numqueries` arm B vs `mflickr_num_queries_pooled` N=1,
0.8453 vs 0.8454) — architecturally the *simplest* point in the whole sweep,
where the pooled and concat heads collapse to the same computation. Applying
that number to every other config was an overgeneralization.

Direct counter-evidence, same seed, same architecture, decisively NOT close:
`mflickr_lph_vs_ortho_multiseed` (ortho=0.1, seed=333, num_queries=4,
sub_batch=96, wavelet=haar by default) scored **0.8584**.
`mflickr_wavelet_type_ablation`'s haar arm — audited as override-identical to
that same config (section 5b) — scored **0.8337**. Same seed, same
(audited) overrides, **0.0247 apart**. That is 14x the N=1 reproducibility
figure and comparable to several effects reported in this document.

Investigated in depth (2026-08-17) — six candidates, checked against the actual
code rather than assumed:

1. **The `hub_utils.py` pin itself — RULED OUT.** `torch.hub`'s own ref
   resolution (`_parse_repo_info`), when no branch is given, opens
   `github.com/.../tree/main/` specifically to check whether `main` exists and
   uses it if so. dinov2 has a `main` branch, so the OLD unpinned call and the
   NEW `facebookresearch/dinov2:main` pin resolve to the exact same ref. The
   commit `c2469dc` change (2026-08-12) only removes an extra network
   round-trip and adds `skip_validation` + retries — it does not change which
   version of the code gets fetched. This was the leading suspect and it
   doesn't hold up.
2. **`main` is a branch, not a commit — REAL, but pre-existing and NOT caused
   by the fix above.** Both the old and new code fetch whatever is at HEAD of
   `facebookresearch/dinov2:main` into an empty Colab cache each fresh
   session, pinned or not. If upstream pushed anything to `main` between the
   two runs' dates, the model code (not the pretrained weights, which come
   from a static URL) could genuinely differ. A web search for recent
   activity on `dinov2/layers/attention.py`, `block.py`, `vision_transformer.py`
   found no evidence of active development — Meta's attention on this repo has
   largely moved to DINOv3 — so this is possible but not the likeliest cause.
   Still worth closing permanently: see the fix below.
3. **In-process sequential Hydra jobs vs. independent subprocesses — RULED
   OUT.** The reference study is one `sweep` (6 jobs run sequentially inside a
   single Python process via Hydra's default in-process sweeper); the wavelet
   study uses `sweep_zip`, which `run_plan.py` dispatches as 3 separate
   `subprocess.run` calls. This looked like a real structural difference, but
   `run.py:59-65` reseeds `random`/`numpy`/`torch`/`torch.cuda` and resets
   `cudnn.deterministic`/`benchmark` at the top of *every* job — in-process or
   not, nothing carries over between jobs through those flags.
4. **`PYTHONHASHSEED`-driven ordering — RULED OUT for MIRFLICKR.** Never
   pinned anywhere in the repo, so it's randomized per-process by default —
   real risk in general if any dataset code enumerates a `set()` for label
   order. Checked `MIRFlickrHashing`: labels come from fixed-column
   `train.txt`/`test.txt` files parsed in file order, `instance_dict` is a
   `defaultdict` keyed by integer class index built by enumeration, not by any
   hash-order-dependent structure. Not the cause here, though the same check
   should be redone for any dataset that does build a vocabulary from a `set`.
5. **Different GPU / driver / package versions across separate Colab
   sessions — PLAUSIBLE, UNVERIFIABLE RETROACTIVELY.** Colab doesn't guarantee
   the same GPU model between sessions, and nothing in this repo pins or logs
   `torch`/CUDA/cuDNN versions. A different GPU can select different cuDNN
   convolution kernels even under `deterministic=True` (the *set* of available
   deterministic algorithms differs by architecture), and a different library
   version can change numerics outright. No way to check this after the fact
   for the two specific runs already done.
6. **Genuine same-seed stochasticity at this architecture's scale — the
   remaining, most likely candidate.** `torch.use_deterministic_algorithms(True)`
   is not set, so atomicAdd-based backward kernels stay nondeterministic
   regardless of `cudnn.deterministic`. At N=1 (the trivial config) this
   compounds to ~0.0001 over 50 epochs; at N=4 with the full cross-attention +
   ortho loss, far more surface area exists for it to compound through. The N=1
   measurement may simply have picked the one config where it doesn't show.

**Fix applied so this stops being unanswerable**: `run.py` now logs
`torch`/CUDA/cuDNN versions, the GPU name, and the dinov2 hub cache's resolved
commit (when available) once per job, at the point the seed is set. Every run
from now on carries a checkable environment fingerprint; candidates 2 and 5
above will be diagnosable in one grep instead of reconstructed after the fact.

**Resolved by a natural triplicate (2026-08-17), without needing the planned
rerun.** `mflickr_num_queries_sb96`'s N=4 arm is override-identical to the
same reference config (ortho=0.1, seed=333, sub_batch=96, wavelet=haar
default) and landed on a *third* value: **0.8459**. Three independent runs of
the nominally same config now exist:

| run | maphashing (best epoch) |
|---|---|
| `mflickr_lph_vs_ortho_multiseed` | 0.8584 |
| `mflickr_wavelet_type_ablation` (haar arm) | 0.8337 |
| `mflickr_num_queries_sb96` (N=4 arm) | 0.8459 |

mean 0.8460, **std 0.0124, range 0.0247** (n=3). This doesn't distinguish
which of candidates 2/5/6 is the mechanism (all three studies plausibly ran on
different Colab sessions/dates, so environment drift and algorithmic
nondeterminism are still both live) — but it settles the practical question:
**same-seed reruns of this architecture are not meaningfully more
reproducible than different-seed runs.** 0.0124 is close enough to the
between-seed σ below that "seed" is not actually buying the reproducibility
its presence in a config implies for N≥4. The originally-planned rerun is no
longer needed to make this actionable.

**Between-seed variance: σ ≈ 0.0174, and now corroborated by an independent
same-seed measurement (0.0124, above) instead of resting on one study.** From
the only properly multi-seeded study (`mflickr_lph_vs_ortho_multiseed`, 3
seeds × 2 ortho settings): the ortho effect itself (+0.0028 ± 0.0055, n=3) is
*not* distinguishable from this between-seed noise (section 1). Every other
study in this log — num_queries, wavelet type, lambda2 — is single-seed, so a
difference smaller than roughly **2σ ≈ 0.025–0.035** between two arms of a
single-seed sweep cannot be told apart from "we happened to draw seed 333"
without rerunning at other seeds. The wavelet-type spread (0.0197 between
bior4.4 and db4, section 5b) sits inside that band — not resolved, needs more
seeds if it's going to be more than a sensitivity curve in the paper.

**What this means for the paper, given the compute budget is fixed and a
paper has to go out regardless:**

1. Say what each number actually is. A single-seed sweep is a *sensitivity
   analysis* — it answers "is the method brittle to this choice", which is
   what R2 literally asked for. It is not a significance-tested comparison.
   Label it that way in the text and in table captions instead of implying
   more precision than n=1 supports.
2. Reserve "X beats Y" language for the one claim that has multi-seed
   backing (orthogonality — and there the honest finding is *no significant
   gain*, which is still a defensible, citable result: R3 asked whether ortho
   regularization does what it's supposed to, and the mechanistic answer is
   yes for the geometry, no measurable effect on mAP).
3. For every single-seed number reported (wavelet type, num_queries,
   lambda2), state the noise floor once, next to the table, as the reader's
   yardstick — σ ≈ 0.012–0.017 (both same-seed reruns and different-seed runs
   land in this band), so "differences below ~0.02–0.03 should be read as
   directional, not conclusive." That one sentence pre-empts the reviewer
   question instead of inviting it.
4. Where a single-seed result changes the paper's actual claims (not just a
   sensitivity curve) — e.g. if wavelet type ends up in the abstract — it
   needs the 2-3 extra seeds before submission; where it's just "here's the
   shape of the curve, the method isn't fragile," n=1 with the caveat above is
   normal practice and matches what R2 asked for.
5. **One canonical number for the default config, cited everywhere.** Three
   different runs of "seed=333, ortho=0.1, num_queries=4, sub_batch=96" now
   exist in this project (0.8584 / 0.8337 / 0.8459) because each ablation
   independently reran the default arm as its own reference point. If two of
   those land in two different paper tables, a reviewer *will* notice the
   mismatch before section 0 explains it. Fix: pick one number for "the
   default config" — the multi-seed mean from `mflickr_lph_vs_ortho_multiseed`
   (0.8401, section 1) is the right choice, since it's already averaged over
   seeds — and reuse that exact number as the reference row in every other
   table (wavelet type, num_queries, lambda2). Do not silently swap in
   whichever single-seed number a given ablation study happened to produce.

**Paragraph to adapt for the paper's Experimental Setup / Reproducibility
subsection**, pre-registered here so it can't be written post-hoc to fit
whatever number comes out later:

> All experiments use `cudnn.deterministic=True` and fixed seeds for Python,
> NumPy and PyTorch. This does not guarantee bit-exact reproducibility: PyTorch
> only fully removes GPU-side nondeterminism under
> `torch.use_deterministic_algorithms(True)`, which we did not enable for
> training-speed reasons, so atomicAdd-based backward kernels retain a small
> amount of run-to-run variance. We measured this directly rather than
> assuming it away: three independent runs of the default configuration
> (seed=333) produced maphashing_level0 ∈ {0.8337, 0.8459, 0.8584}
> (σ = 0.012), comparable to the between-seed standard deviation measured over
> 3 seeds (σ = 0.017, Table X). We therefore report the default configuration
> as a mean over 3 seeds throughout, and treat differences smaller than
> ≈0.02–0.03 in single-seed sensitivity analyses (Sections Y, Z) as
> directional rather than conclusive.

This is why section 1's honest conclusion ("not distinguishable from noise")
is itself useful ammunition: it shows the noise floor was measured, not
assumed, which is the actual gap R2/R3 flagged in the original submission.

## 1. Orthogonality: no significant gain (R2)

Paired by seed, best epoch:

| seed | ortho=0.0 | ortho=0.1 | diff |
|---|---|---|---|
| 111 | 0.8395 | 0.8382 | −0.0013 |
| 222 | 0.8230 | 0.8237 | +0.0007 |
| 333 | 0.8494 | 0.8584 | +0.0090 |

Mean paired diff **+0.0028 ± 0.0055** (n=3). Sign flips at seed 111; std ≈ 2×
the mean; paired t ≈ 0.88, p ≈ 0.47. On `map_level0` the same computation gives
+0.0017 ± 0.0076, p ≈ 0.73.

**The mAP gain is not distinguishable from noise and cannot be claimed**, as a
headline result or as a minor one.

## 2. No collapse of any kind (R3)

Query orthogonality behaves exactly as designed — but there was little to fix:
the un-regularized arm is already near-orthogonal (mean |off-diag| 0.042),
and the regularized one is exact (0.0000).

Attention shows no collapse at all, on either dataset, with or without ortho:
per-query entropy sits at log(4) = 1.386, the theoretical maximum, and the LL
share across the 8 heads has std ≈ 0.003.

| | MIRFLICKR ortho=0.0 | MIRFLICKR ortho=0.1 | VOC ortho=0.1 |
|---|---|---|---|
| attention share LL / LH / HL / HH | 0.258 / 0.252 / 0.247 / 0.242 | 0.258 / 0.250 / 0.250 / 0.243 | 0.265 / 0.247 / 0.242 / 0.246 |
| per-query entropy | 1.386 (max) | 1.386 (max) | 1.385–1.386 (max) |
| band read-out cosine (mean off-diag) | 0.716 | 0.682 | 0.465 |

The original figure showing total LL dominance came from `scores * 10.0`
applied before the softmax, against PyTorch's true `1/sqrt(head_dim)` = 1/√48 —
a **69× logit inflation**. Passing the measured shares through that same factor
reproduces it almost exactly (LL=0.875, entropy 0.66 bits). The *direction*
(LL mildly preferred) is real; the *magnitude* is negligible.

VOC is the stronger case: there, LL is nearly orthogonal to the detail bands
(read-out cosine 0.139), so there genuinely *is* something to differentiate —
and the attention still doesn't. Consistent across both datasets: LH/HL/HH are
largely redundant with each other (0.78–0.87) while LL stands apart.

## 3. Root cause: direction is constrained, magnitude is not

| | value |
|---|---|
| query_token norm at init (`trunc_normal_` std=0.02, dim 384) | 0.3919 |
| measured after 40 epochs | 0.3775 – 0.4047 |
| growth ratio | **1.009** |
| `\|\|W_q q_i\|\| / \|\|b_q\|\|` | 11.13 |
| level-2 (projected query) off-diag cosine | 0.2124 |
| pre-softmax within-row score spread | 0.0952 |

`compute_ortho_loss` normalizes Q before the Gram matrix, so it is exactly
scale-invariant and exerts zero pressure on `||q||`; AdamW `weight_decay=0.0005`
shrinks it; `lr=1e-5`. The queries therefore never leave their initialization
scale, scores stay tiny, and softmax over a spread of 0.095 is *mathematically
forced* to be near-uniform — predicted LL=0.268 vs measured 0.265.

The earlier "shared bias `b_q` dominates" hypothesis is **refuted** (ratio 11.13,
and the projected queries remain distinct).

Selectivity would need `||q||` of ~2.8 / ~6.2 / ~10.2 for a dominant band at
40% / 60% / 80%. Tested by `mflickr_query_scale_decoupled.yaml`.

## 4. Capacity control (R3): the architecture does earn its keep

| | best-epoch maphashing |
|---|---|
| ViT-B alone (~86M, 1 backbone) | 0.8155 (epoch 40) |
| MBW-DINO ortho=0.0, seed 333 | 0.8494 |
| MBW-DINO ortho=0.1, seed 333 | 0.8584 |
| MBW-DINO mean over 3 seeds | 0.8373 – 0.8401 |

**+2.2 to +4.3 points** for MBW-DINO — roughly 10× the orthogonality effect. The
multi-band decomposition contributes something real beyond raw capacity.

Caveats: single seed; and the two arms did **not** share an augmentation
pipeline (`basic` = plain RandomCrop for ViT-B vs `basic_swt` =
RandomResizedCrop for MBW-DINO). `config/transform/basic_rrc.yaml` and the
backbone-grid studies exist to remove that confound.

The first attempt at this control returned ~79% and was invalid: `DINOHashBaseline`
kept its backbone frozen regardless of the `frozen` flag (`getattr(self.backbone,
'frozen', True)` on an attribute that was never set). Fixed.

## 4b. Forcing attention selectivity changes nothing

`mflickr_query_scale_decoupled` (new `CrossAttentionBottleneckHeadDecoupled`,
`q_effective = query_scale * normalize(query_tokens)`, seed 333, ortho=0.1):

| query_scale_init | targeted regime | best-epoch maphashing |
|---|---|---|
| 2.0 | dominant band ~35% | 0.8364 |
| 6.0 | ~59% | 0.8445 |
| 12.0 | ~86% | 0.8324 |
| — | reference (Advanced head, ortho=0.1, seed 333) | 0.8584 |
| — | reference, mean over 3 seeds | 0.8401 ± 0.0174 |

Spread across the whole sweep: **0.0121**, i.e. smaller than the reference's
seed-to-seed std (0.0174), and non-monotonic — the most selective setting is
the worst. Selectivity is not what the architecture was missing, and the
uniform attention documented in section 2 was costing nothing.

Read together with section 4 (MBW-DINO beats a parameter-matched ViT-B by
2.2–4.3 points), the coherent reading is that the multi-band advantage comes
from the decomposed **inputs** and an ensemble-like combination, not from any
learned routing between bands. The cross-attention bottleneck is not doing the
job it was designed for, at any magnitude.

Not done, by decision (2026-08-12): the follow-up
`measure_attention_collapse.py` pass on these three checkpoints, which would
have confirmed that the attention actually did become selective and shown where
the learnable `query_scale` drifted. Without it, "we gave the model selectivity
and it rejected it" is the plausible reading but not a verified one — state it
accordingly if it goes in the paper.

## 5. SWT transform: correct

Contract holds (`[C=3, S=4, 224, 224]`, `x[:, :, i]` → `[B, 3, H, W]`), the four
sub-bands are genuinely distinct (mean cosine 0.038). Nothing to fix.

Unresolved: no Normalize step, so the detail bands reach DINOv2 far below its
pretraining scale — HH std 0.0242 vs ~1. Whether the patch embedding survives
that has not been measured.

## 5b. Wavelet type sensitivity (R2): bior4.4 > haar > db4

`mflickr_wavelet_type_ablation`, seed 333, ortho=0.1, num_queries=4, sub_batch=96,
level=1, best epoch:

| wavelet | best-epoch maphashing | epoch | final (epoch 50) |
|---|---|---|---|
| bior4.4 | **0.8534** | 30 | 0.8478 |
| haar (reference) | 0.8337 | 35 | 0.8215 |
| db4 | 0.8286 | 50 | 0.8286 (= best, monotonic) |

haar is the reference config, not the worst arm — db4 is. bior4.4 (JPEG2000's
wavelet, longer filter support, smoother detail bands) beats it by +0.0197.
Single seed; the same-seed noise floor for this exact architecture is now
measured directly at σ ≈ 0.0124 (section 0), so this spread is inside the
noise band and not yet distinguishable from it — needs more seeds before
being read as a real ranking.

haar's trajectory is also visibly noisier and less monotonic than the other
two (bit_balance 0.43-0.60 vs bior4.4/db4's tighter 0.31-0.51), and its final
epoch (0.8215) is its worst point in the last 15 epochs — the reverse of
db4, whose final epoch is its best. Worth a second seed before reading
anything into the ranking.

## 5c. num_queries sensitivity, redone clean at sub_batch=96 (R2)

`mflickr_num_queries_sb96`: same reference config, sub_batch fixed to 96 for
every arm (no Ghost-BN-path confound), `fusion_config.dropout` restored to
its model-default 0.1 (an earlier study had silently overridden it to 0 —
fixed here, see the study's own header comment). Seed 333, ortho=0.1,
best epoch:

| num_queries | best-epoch maphashing | epoch |
|---|---|---|
| 1 | 0.8460 | 40 |
| 2 | 0.8274 | 30 |
| 4 | 0.8459 | 25 |
| 8 | 0.8445 | 45 |

This is a materially different picture from the earlier sub_batch=48 curve
(`mflickr_num_queries_ablation`: N=1=0.8802 > N=2=0.8652 > N=4=0.8503 >
N=8=0.8308, monotone decreasing). Here, N=1, N=4 and N=8 are statistically
indistinguishable from each other (spread 0.0015, well inside the σ≈0.0124
same-seed noise floor measured in section 0) — N=4 (0.8459) is effectively
identical to N=1 (0.8460), not 3.5 points behind it. Only N=2 sits out
(0.8274, ~18 points below the other three, ~1.5σ), and even that is
suggestive rather than conclusive on n=1.

**This confirms the suspicion raised when the sub_batch confound was first
found**: most of the earlier "fewer queries is better" narrative was a
sub_batch=48 (Ghost-BN-like) effect riding on top of num_queries, not a
genuine property of the architecture. Once sub_batch is held at 96 throughout,
num_queries looks close to flat from 1 to 8, with N=2 as the one point that
doesn't fit — worth a rerun before deciding if that's real or noise.

**Scope decision (2026-08-17): num_queries is not presented as a
contribution.** Consistent with the data above and with section 4b
(forcing attention selectivity with `query_scale` didn't help either) — there
is no evidence anywhere in this project that the number of queries, or what
they individually attend to, does anything functionally useful. `num_queries=4`
was originally motivated as "4 experts to separate the 4 subbands"
(`config/model/multidino_attention_hashing_ortho.yaml`'s own comment); that
motivation is now unsupported on two independent fronts (attention never
specializes by band, section 2; and N=1/4/8 perform equally, this section).
The paper reports this as a **sensitivity/robustness result** — the method is
insensitive to `num_queries`, which is itself a legitimate answer to R2 — and
does not claim `num_queries=4` or any other value as a design choice earned
by the ablation. If efficiency is worth a sentence: N=1 reaches the same mAP
with a narrower `out_proj` (`Linear(384,384)` vs `Linear(1536,384)` at N=4),
which is a defensible minor remark, not a contribution.

## 5d. Final headline results (revision numbers for the paper's Table 1)

One seed (333) per (dataset, nbits), `num_queries=1`, `ortho_weight=0.1`, `sub_batch=96`, `basic_swt`, zero evaluation during training (all eval is post-hoc via `evaluate_all_checkpoints.py`), matching the rejected paper's own bit-length protocol per dataset. See `mflickr_final_headline.yaml` / `voc_final_headline.yaml` / `coco_final_headline.yaml`.

MIRFLICKR-25K (`--k 19581`, hamming, `maphashing_level0`, best epoch of {5,10,...,50}):

| bits | best epoch | mAP |
|---|---|---|
| 32 | 35 | 0.8110 |
| 64 | 40 | 0.8506 |
| 128 | 30 | 0.8461 |

64 bits slightly beats 128 bits here — inside the noise band established in section 0 (σ≈0.012-0.017), not a claim that more bits hurt.

VOC 2012 and MS COCO: not yet evaluated. VOC needs a single `--k 5717` pass (mAP@ALL only, per the paper's protocol for that dataset). COCO needs `--k 5000,117218` (mAP@5000 **and** mAP@ALL in one pass, via the new multi-k path in `evaluate_multi_k()` — see `main/engine/evaluate.py` — so the checkpoint's embeddings aren't recomputed twice). 117218 is `coco_database.txt`'s confirmed line count.

## 6. Method fixes made along the way

- `build_fast_eval_subset` grouped multi-hot label tensors by identity hash → empty subset → `UnboundLocalError`. Fixed to use `dataset.instance_dict`.
- fast_eval is **not** a reliable proxy: it stayed ~0.95 while the test split was fully collapsed (bit_balance 0.0, frozen mAP 0.7736). Kept as a divergence canary only; epoch selection goes through `evaluate_all_checkpoints.py`.
- DataLoader workers were unseeded → genuine run-to-run non-determinism (one run collapsed, an identical rerun did not). Fixed with `worker_init_fn` + `generator`; `clip_grad: 5.0` enabled.
- `DINOHashBaseline` never fine-tuned its backbone (above).
- `SingleBandNet` applied its own `tanh` on top of `HashLoss`'s (`tanh(tanh(x))`, bounded at 0.762, so the quantization term could never fall below 0.238) and had no BatchNorm, unlike the reference. Aligned.
- `SCHLoss` assumes codes already in [-1, 1]; added an opt-in `apply_tanh` for raw-logit models (default off, existing configs untouched).

## 7. Open questions

- **Is ~77% the floor on MIRFLICKR@all?** A run with all 64 bits dead (bit_balance 0.0) still scored 0.7736. Published 64-bit results on this benchmark span ~0.65–0.85, so the headroom may be much smaller than it looks. `measure_random_baseline.py` measures it; not yet run.
- **Are the detail bands dead at the patch embedding?** (section 5)
- `mflickr_single_band_ablation` returned ~82% for every band — needs re-reading after the double-tanh/BN fix.
- Seeds 111/222 not yet passed through the corrected attention diagnostic; no VOC ortho=0 arm.
