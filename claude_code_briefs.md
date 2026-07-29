# Briefs for Claude Code — two remaining ACIVS 2026 revision experiments

Context: MBW-DINO is a deep hashing retrieval model under major revision.
Reviewer #3 asked for (1) a parameter-matched control isolating the
wavelet/frequency contribution from the raw capacity of using 4 parallel
DINOv2 branches, and (2) direct evidence that the orthogonality
regularization on the cross-attention queries actually reduces "attention
collapse," rather than inferring it only from downstream mAP. Both briefs
below are self-contained — paste each one into Claude Code separately, or
both in sequence.

Repo root: `C:\These\image retrieval\Retrieval Framework`
Config system: Hydra. Training studies are defined as yaml files under
`studies/`, with a `base_overrides` block (static Hydra overrides) and a
`sweep` block (cartesian product of swept keys), e.g. see
`studies/voc_num_queries_ablation_multidino_attention_hashing_ortho.yaml`
for the exact format to copy.

Already verified this session (don't re-derive, just use):
- The model's fusion head is `CrossAttentionBottleneckHeadAdvanced` in
  `main/models/multi_dino_attention.py` (~line 763), selected via
  `fusion_config.type: cross_attention_advanced` in
  `config/model/multidino_attention_hashing_ortho.yaml`.
- The model uses 4 unshared `dinov2_vits14` backbones
  (`backbones_config` list of 4 identical entries, `frozen: False` each).
- Real hyperparameters used to produce the paper's headline results:
  `ortho_weight: 0.1` (not the in-repo default of 0.01), `quant_weight: 0.1`
  (from `config/loss/hash_loss.yaml`), `num_queries: 4`.
- `num_classes: 20` for VOC (`config/loss/hash_loss.yaml`); do NOT use 38
  (that's MIRFLICKR's tag count — a bug we already found and fixed once in
  `voc_num_queries_ablation_...yaml`, don't reintroduce it).
- VOC dataset config: `config/dataset/voc.yaml`, `data_dir: /content/voc2012`
  (Colab path — do not point it at `/content/data/mirflickr`).
- Wavelet transform: `config/transform/voc_swt.yaml`, applies `SWTTransform`
  (level 1, Haar) as the last step of both train and test pipelines, after
  `Resize`/`RandomResizedCrop`/`ColorJitter`/`RandomHorizontalFlip` (train)
  or `Resize`/`CenterCrop` (test). Find the `SWTTransform` class
  implementation (search the repo for `class SWTTransform`) to see exactly
  what it outputs (likely 4 subbands stacked or listed) before touching the
  data pipeline.

---

## Brief 1 — Parameter-matched control ("Config A: Same-Image ×4")

**Goal.** Isolate the contribution of the wavelet/frequency decomposition
from the raw effect of using 4 parallel DINOv2 branches instead of 1. This
is the central ask from Reviewer #3 and the highest-priority remaining
experiment for this revision.

**What to build.** A new model/transform configuration where the 4 DINOv2
branches receive 4 identical copies of the same raw (non-wavelet-decomposed)
224×224 image, instead of the 4 SWT subbands (LL, LH, HL, HH). Everything
else (4 unshared `dinov2_vits14` backbones, `CrossAttentionBottleneckHeadAdvanced`
fusion with `num_queries=4`, `ortho_weight=0.1`, LPH hashing head,
`quant_weight=0.1`) must stay identical to the real MBW-DINO config, so the
parameter count and architecture are exactly matched — only the input
content differs.

**Suggested approach.**
1. Find the `SWTTransform` class and the transform pipeline entry point that
   currently produces the 4-subband tensor consumed by the model.
2. Add a new transform (e.g. `IdenticalCopyTransform` or a flag on
   `SWTTransform` like `identity_mode: true`) that, instead of computing SWT
   subbands, just duplicates the single transformed image 4 times into the
   same output shape/format the model already expects. Keep the same
   `Resize`/crop/augmentation steps upstream — only the SWT step itself
   should be bypassed.
3. Add a new transform config, e.g. `config/transform/voc_identity4.yaml`,
   mirroring `voc_swt.yaml` but using the new identity-copy step.
4. Add a new study yaml, e.g.
   `studies/voc_paramcontrol_sameimage_multidino_attention_hashing_ortho.yaml`,
   copied from `studies/voc_num_queries_ablation_multidino_attention_hashing_ortho.yaml`
   as a template, with these overrides:
   - `transform: voc_identity4` (the new transform config)
   - `dataset: voc`, `dataset.kwargs.data_dir: /content/voc2012` (verify this
     is correct, don't inherit the mirflickr path bug)
   - `loss.0.kwargs.num_classes: 20`
   - `model.kwargs.fusion_config.ortho_weight: 0.1`
   - `model.kwargs.fusion_config.num_queries: 4`
   - `model.kwargs.binary_config.nbits: 64`, `loss.0.kwargs.embedding_size: 64`
   - `sweep: experience.seed: [111, 222, 333]` (3 seeds, matching the other
     multi-seed studies already run for this revision)
5. Run it and report mAP@ALL on VOC 2012 (64 bits), mean ± std over the 3
   seeds.

**Interpretation to report afterward** (not for Claude Code to write, just
context): compare this "Same-Image ×4" result against the existing
single-branch baseline (94.31, from `studies/.../bn_ablation` or the
original Table 2) and the real MBW-DINO result (98.20 / 98.68 depending on
config). If Same-Image ×4 recovers most of the gap, the backbone's gain is
mostly capacity; if it stays close to the single-branch baseline, the gain
is attributable to the wavelet decomposition.

**Do not** modify `voc_num_queries_ablation_...yaml`,
`voc_lambda_2_ablation_...yaml`, or `bn_ablation_voc.yaml` — only add new
files.

---

## Brief 2 — Attention-collapse diagnostic (Gram matrix + query-to-subband attention)

**Goal.** Provide direct, mechanistic evidence that the orthogonality
regularization reduces query redundancy, rather than relying only on
downstream mAP (Reviewer #3's specific complaint).

**No new training needed if checkpoints exist.** Check whether
`studies/voc_lambda_2_ablation_multidino_attention_hashing_ortho.yaml`
already has saved checkpoints (under `experience.log_dir: ./experiments_runs`,
look for a `.hydra/config.yaml` + model checkpoint per run) for
`ortho_weight = 0` and `ortho_weight = 0.1` (or whatever the current real
default is, per Part 0 of `../revision_report_and_plan.md`). If so, reuse
those checkpoints directly instead of retraining.

**What to build.** A small standalone analysis script (e.g.
`scripts/attention_diagnostics.py`) that:
1. Loads a trained `CrossAttentionBottleneckHeadAdvanced` checkpoint (one
   trained with ortho_weight > 0, one trained with ortho_weight = 0).
2. Computes the query Gram matrix: `Q = self.query_tokens.squeeze(0)`,
   `Q_norm = F.normalize(Q, p=2, dim=-1)`, `gram = Q_norm @ Q_norm.T`
   (this mirrors the existing `compute_ortho_loss` method in
   `multi_dino_attention.py` — reuse that logic, don't reimplement it from
   scratch).
3. Reports the mean off-diagonal magnitude of `gram` (mean pairwise cosine
   similarity between the 4 queries) for both checkpoints — this is the
   single number to put in the paper (e.g. "0.87 → 0.05").
4. Optionally, saves the 4×4 Gram matrix as a heatmap image (`imshow`) for
   both checkpoints, side by side, for a small figure if page budget allows.
5. Optionally (higher value, more work): runs a forward pass on a batch of
   validation images and captures `attn_weights` from the
   `self.attn(query=q, key=kv, value=kv)` call in `forward()` (currently
   discarded — check if it needs `need_weights=True` on the
   `nn.MultiheadAttention` call, and whether weights need averaging over
   heads/batch), then reports, per query, the average attention mass placed
   on each of the 4 subband tokens (LL/LH/HL/HH) — with vs without
   orthogonality. Present as a 4×4 table or small heatmap: "query i →
   subband j" attention mass.

**Acceptance criteria:** at minimum, one printed/saved number (off-diagonal
Gram mean, with vs without ortho) for VOC 64-bit checkpoints. The
attention-mass breakdown (step 5) is a stretch goal if time allows — it's
the most convincing single piece of evidence for the paper but not required
to unblock the revision.

**Do not** modify the training code or loss computation — this is a
read-only analysis over existing/trained model weights.
