# Results log — MIRFLICKR / VOC revision

Everything measured during the ACIVS26 revision, kept in one place so no number
has to be recovered from a chat log. Raw outputs live beside this file; the
interpretation and the experiment plan live in `../MIRFLICKR_DIAGNOSTIC_PLAN.md`.

| file | what it is |
|---|---|
| `lph_vs_ortho_multiseed_per_epoch.csv` | 6 runs (ortho 0.0/0.1 × seeds 111/222/333), every saved epoch, from `evaluate_all_checkpoints.py` |
| `vitb_capacity_control_per_epoch.csv` | ViT-B capacity control, every saved epoch |
| `diagnostics_attention_2026-08-11.txt` | verbatim `measure_query_orthogonality.py` + `measure_attention_collapse.py` output (MIRFLICKR + VOC) |
| `swt_transform_check_2026-08-12.txt` | verbatim `verify_swt_transform.py` output |

All mAP figures below are `maphashing_level0` (the hashing-literature mAP@topk
from `calculate_maphashing`), best epoch per run, `top_k=19581`, hamming.
`map_level0` (torchmetrics `RetrievalMAP`) is also in the CSVs and occasionally
picks a different best epoch — the two disagree on 2 of the 6 multiseed runs.

---

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

## 5. SWT transform: correct

Contract holds (`[C=3, S=4, 224, 224]`, `x[:, :, i]` → `[B, 3, H, W]`), the four
sub-bands are genuinely distinct (mean cosine 0.038). Nothing to fix.

Unresolved: no Normalize step, so the detail bands reach DINOv2 far below its
pretraining scale — HH std 0.0242 vs ~1. Whether the patch embedding survives
that has not been measured.

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
