# Guide de rédaction — révision ACIVS26 (MBW-DINO hashing)

Ce fichier donne, section par section, quoi changer par rapport au papier rejeté,
quelle structure adopter, et des formulations d'exemple (en anglais, langue du
papier) à adapter dans vos propres mots — ce sont des points de départ
rhétoriques, pas du texte à copier-coller. Chaque section renvoie aux données
exactes dans `studies/results/RESULTS.md` (section indiquée entre parenthèses).

---

## Titre / Abstract

**Contrainte : le papier accepté n'est pas cité (décision des encadrants).**
Pas de mention, directe ou indirecte, de ce papier dans le texte. Ça retire
l'option "désambiguïser via citation" — la seule solution qui reste est de
ne pas réutiliser le nom du tout.

- Titre : garder le titre original du papier rejeté (*"A Multi-Frequency Deep
  Hashing Retrieval..."*) plutôt que "MBW-DINO" — il est déjà différencié et
  décrit la vraie contribution (hashing, pas continuous retrieval).
- Ne pas utiliser "MBW-DINO" dans le corps du texte de la révision non plus,
  même sans citation — le nom seul, combiné aux mêmes auteurs et à une
  architecture identique, suffit à créer un rapprochement visible une fois
  les deux papiers publiés. Utiliser une description neutre à la place :
  "our multi-band wavelet-DINOv2 hashing architecture".

**Abstract** : ne pas promettre une contribution que les données ne
soutiennent plus. Éviter "we propose a query-based routing mechanism that
selects relevant frequency bands" (l'ancienne formulation implicite). Formuler
plutôt sur ce qui tient (section 8 de RESULTS.md) :
> "We show that decomposing images into wavelet sub-bands and processing them
> through independent DINOv2 branches improves hashing retrieval over a
> parameter-matched single-branch baseline by X points, and that this gain
> does not rely on learned attention selectivity between bands — a claim we
> verify directly rather than infer from ablation deltas."

---

## 1. Introduction

**Structure recommandée** (4 paragraphes) :
1. Contexte : CNN → ViT → SSL foundation models (DINO/DINOv2) pour la
   représentation visuelle.
2. Gap identifié : les ViT classiques (pas SSL) ont été utilisés pour le
   hashing ; le pretraining SSL et l'information multi-fréquence restent
   sous-exploités dans ce cadre précis.
3. Votre proposition : SWT + branches DINOv2 indépendantes + fusion par
   cross-attention, entraîné avec une loss à base de proxy pour le hashing.
4. Résumé des contributions — **à réécrire pour matcher ce qui est prouvé**,
   pas ce qui était espéré à l'origine.

**Formulation pour le fil narratif** (reprend votre propre résumé) :
> "While Vision Transformers have progressively replaced CNNs as backbones for
> deep hashing, most prior work relies on classification-pretrained ViTs
> rather than self-supervised foundation models such as DINOv2. We combine
> DINOv2 with a stationary wavelet decomposition to inject multi-frequency
> structural priors into the hashing pipeline, and train with a proxy-based
> hashing loss — while noting our architecture is loss-agnostic."

**Liste de contributions — version révisée**, dans l'ordre où elles sont
réellement soutenues par les données :
1. Adaptation de la décomposition SWT multi-bandes + branches DINOv2 au
   hashing profond, avec un gain mesuré contre un baseline ViT à paramètres
   égalisés (section 4, +2.2 à +4.3 pts).
2. Une étude de sensibilité exhaustive (R2) : type d'ondelette, nombre de
   queries — répondant explicitement à la demande de robustesse des reviewers.
3. Une analyse mécanistique directe (R3) du module de fusion — pas seulement
   des deltas d'ablation — montrant que le gain vient de la diversité des
   entrées, pas d'un routing appris (sections 2-3-4b).
4. Une simplification architecturale justifiée empiriquement : N=1 query
   suffit, éliminant le besoin de régularisation d'orthogonalité (section 5c).

Ne mettez PAS "we introduce an orthogonality-regularized multi-query
attention mechanism that learns to route between frequency bands" dans la
liste de contributions — c'est exactement l'affirmation invalidée.

---

## 2. Related Work

**Deux sous-sections à garder**, structure similaire au papier accepté (c'est
un choix de structure légitime à réutiliser, seul le TEXTE doit être
réécrit) :

### 2.1 Wavelets in deep learning / deep hashing
**Le papier accepté n'est pas cité** (décision des encadrants) — ne pas s'y
référer, même indirectement ("in a related work by the same authors...").
Citer vos travaux antérieurs qui restent pertinents et publiés/publiables
indépendamment (ComNet 2025, SIVP 2025, Pattern Recognition 2023) pour
ancrer la lignée wavelet+deep learning du groupe, sans mentionner le papier
MBW-DINO continuous-retrieval lui-même.

### 2.2 Deep hashing baselines
Positionner par rapport aux méthodes de hashing établies (proxy-based losses,
familles DPSH/HashNet/CSQ) — c'est la partie qui ancre ce papier dans sa
propre littérature, distincte de la littérature continuous-retrieval, et
qui n'a pas besoin de mentionner l'autre papier pour être complète.

---

## 3. Method

**Priorité n°1 : ne pas copier-adapter le texte du papier accepté pour
décrire SWT / le module de cross-attention.** Même architecture, texte à
réécrire entièrement dans vos mots, idéalement en la décrivant sans avoir le
PDF du papier accepté ouvert à côté (paraphrase de mémoire plutôt qu'édition
sur place, pour éviter de reproduire inconsciemment sa structure de phrase).
Sans citation possible pour légitimer une ressemblance, c'est la mitigation
la plus importante du papier entier.

**Structure** :
1. Décomposition SWT (bref, description autonome — pas de renvoi vers le
   papier accepté, il n'est pas cité).
2. Branches DINOv2 indépendantes.
3. Module de fusion — décrire la mécanique (Q/K/V, query global appris) sans
   affirmer qu'il fait du "routing" ou de la "sélection". Formulation neutre :
   > "A learnable query token attends over the four sub-band representations
   > via multi-head cross-attention, producing a fused embedding that is
   > passed to the hashing head."
   Pas : "...allowing the model to selectively route to the most informative
   frequency bands" (c'est la phrase à éviter partout dans ce papier).
4. Loss de hashing : proxy-based (`HashLoss`) + régularisation
   d'orthogonalité pour N>1 queries, en notant explicitement qu'elle est
   optionnelle/désactivée quand N=1 :
   > "For N>1 query tokens, we additionally apply an orthogonality
   > regularizer on the queries (Section 5c); this term is vacuous when N=1,
   > the configuration adopted for our main results (Section 5d)."
5. Préciser que l'architecture est agnostique à la loss (comme vous l'avez
   dit) — une phrase suffit, ne pas développer une comparaison de losses non
   testée.

---

## 4. Experimental Setup

Reprendre tel quel ce qui est déjà validé : datasets (MIRFLICKR-25K, VOC2012,
COCO), protocole de bits par dataset (Table 1, RESULTS.md section 5d),
hyperparamètres (AdamW lr=1e-5, wd=5e-4, CosineAnnealing).

**Paragraphe de reproductibilité — à insérer tel quel** (déjà rédigé et
validé, RESULTS.md section 0) :
> "All experiments use `cudnn.deterministic=True` and fixed seeds for Python,
> NumPy and PyTorch. This does not guarantee bit-exact reproducibility:
> PyTorch only fully removes GPU-side nondeterminism under
> `torch.use_deterministic_algorithms(True)`, which we did not enable for
> training-speed reasons, so atomicAdd-based backward kernels retain a small
> amount of run-to-run variance. We measured this directly rather than
> assuming it away: three independent runs of the default configuration
> (seed=333) produced maphashing_level0 ∈ {0.8337, 0.8459, 0.8584}
> (σ = 0.012), comparable to the between-seed standard deviation measured
> over 3 seeds (σ = 0.017, Table X). We therefore report the default
> configuration as a mean over 3 seeds throughout, and treat differences
> smaller than ≈0.02–0.03 in single-seed sensitivity analyses as directional
> rather than conclusive."

Cette phrase préempte directement la critique R2 sur la rigueur statistique —
ne pas la couper pour gagner de la place, c'est probablement la phrase la
plus importante du papier vis-à-vis des reviewers.

---

## 5. Results

### 5.1 Comparison with state of the art / headline table
Table 1 revisitée avec les nombres de la section 5d de RESULTS.md :
MIRFLICKR fait (0.811/0.851/0.846 pour 32/64/128 bits), VOC fait (avec la
réserve sur 32/96 bits, voir plus bas), COCO fait (avec l'explication du mAP
quasi-plafond).

**Pour VOC** — ne pas juste mettre le nombre "best epoch" sans commentaire :
> "The 32-bit and 96-bit VOC arms reach their best epoch early (epoch 10 and
> 5 respectively) before degrading; we report [either the epoch-50 value for
> consistency with the 64-bit arm, or the best-epoch value with this caveat
> stated in a footnote — decide before writing]."
Cette décision (best-epoch vs epoch-50) doit être prise UNE fois et appliquée
uniformément aux trois configurations bits, pas au cas par cas.

**Pour COCO** — expliquer le mAP élevé plutôt que le laisser paraître
suspect :
> "COCO's any-shared-label relevance criterion, combined with a database
> where the single most frequent category covers 54.6% of images, yields
> near-ceiling mAP that saturates well before rank 5000 — consistent with
> known properties of this benchmark family (also observed on NUS-WIDE), not
> a leakage artifact (verified: query/database path sets are disjoint)."

### 5.2 Baselines (ViT-B, ViT-S)
Table de la section 4 : ViT-B seul (0.8155), MBW-DINO ortho=0.0 (0.8494),
ortho=0.1 (0.8584), moyenne 3 seeds (0.8373–0.8401).
> "MBW-DINO improves over a parameter-matched single-branch ViT-B baseline by
> +2.2 to +4.3 points (Table X), indicating that the multi-band decomposition
> contributes representational capacity beyond what raw backbone capacity
> alone provides."
Mentionner la réserve sur le pipeline d'augmentation non partagé entre les
deux arms (section 4's caveat) — une phrase, pas un paragraphe.

### 5.3 Wavelet family study
**Fait (2026-08-19).** Les runs `_nq1` (db4, bior4.4, num_queries=1) sont
terminés sur MIRFLICKR et VOC — les deux datasets donnent des classements
différents entre eux, et différents aussi de l'étude N=4 d'origine de ce
projet. Pas de papier accepté à mentionner ici : l'argument tient déjà sur le
seul désaccord MIRFLICKR-vs-VOC, pas besoin d'un point de comparaison
externe.

> "We do not observe a consistent wavelet-family ranking: MIRFLICKR favors
> db4, VOC favors Haar, and neither matches our earlier num_queries=4 study
> (bior4.4 best). Every observed spread (≤0.005 mAP) is within the measured
> noise floor (σ≈0.012-0.017, Section X). We therefore report insensitivity
> to wavelet-family choice as the robustness finding, rather than endorsing
> any particular family — a result that itself answers R2's sensitivity
> request without requiring the ranking to be resolved."

Chiffres exacts : MIRFLICKR db4 (0.8468) > haar (0.8460) > bior4.4 (0.8423) ;
VOC haar (0.9922) > bior4.4 (0.9893) > db4 (0.9875).

### 5.4 num_queries study + orthogonality study
C'est ici que l'arc "N=4 avec ortho → ça ne marche pas → N=1 suffit" se
déploie (reprend l'échange précédent) :

> "We initially motivated a multi-query design (N=4, one query per sub-band)
> regularized toward orthogonality to encourage specialization. We find this
> regularization achieves its geometric objective (near-exact query
> orthogonality, Section X) but yields no significant mAP gain (+0.0028 ±
> 0.0055 over 3 seeds, p≈0.47, Table X)."

> "To understand why, we directly diagnose the fusion mechanism rather than
> inferring its behavior from ablation deltas. Attention weights are
> near-uniform across sub-bands for essentially every test image (per-query
> entropy at the theoretical maximum log(4); maximum pre-softmax score spread
> across the test set of only 0.336), a pattern explained by the query
> token's magnitude remaining close to its initialization scale throughout
> training (growth ratio 1.009) — the orthogonality regularizer is
> scale-invariant and exerts no pressure on magnitude, while weight decay
> actively shrinks it. Forcing selectivity externally via a decoupled
> magnitude parameter (Section X) does not improve retrieval and is
> non-monotonic with the forced concentration level, confirming that
> selectivity is not what the architecture benefits from."

> "This motivated testing whether the number of query tokens matters at all.
> At N∈{1,2,4,8} with the confound from an earlier microbatching artifact
> removed (Section X), N=1, N=4 and N=8 perform statistically
> indistinguishably (spread 0.0015, within the measured noise floor); only
> N=2 deviates, suggestively but not conclusively on a single seed. We
> therefore adopt N=1 for our main results: it matches the performance of
> more complex configurations with a narrower output projection and without
> requiring orthogonality regularization at all."

C'est le paragraphe le plus important du papier sur le plan argumentatif —
c'est lui qui transforme trois résultats négatifs (ortho n'aide pas,
sélectivité forcée n'aide pas, N ne compte pas) en une histoire cohérente
plutôt que trois échecs disjoints.

### 5.5 Contribution of each sub-band
Reprend l'ablation existante (`mflickr_single_band_ablation`) — vérifiez que
la relecture post double-tanh/BN fix mentionnée en section 7 de RESULTS.md a
été faite avant d'écrire cette sous-section, les nombres actuels (~0.77-0.81)
sont peut-être encore à revoir.

---

## 6. Discussion / Limitations

Point par point, pas de prose générale :
- Études single-seed (wavelet, num_queries) : rappeler le seuil de bruit
  (σ≈0.012–0.017) une fois ici, pas le répéter à chaque table.
- VOC 32/96-bit : l'instabilité en début d'entraînement, si non résolue à la
  soumission.
- Coût computationnel : 4 branches DINOv2 en parallèle — la limitation est
  réelle et se justifie seule (4x le coût d'inférence d'un seul DINOv2-S),
  pas besoin d'un point de comparaison externe pour l'énoncer honnêtement.
- Question ouverte sur le plafond MIRFLICKR@all (~77%, section 7 de
  RESULTS.md) si `measure_random_baseline.py` est lancé à temps.

---

## 7. Conclusion

Reformuler la revendication étroite, sans retomber sur le vocabulaire de
routing :
> "We show that decomposing images into wavelet sub-bands processed by
> independent DINOv2 branches improves deep hashing retrieval over a
> parameter-matched baseline. Contrary to our initial hypothesis, this gain
> does not stem from learned attention-based routing between sub-bands —
> direct diagnostics show the fusion mechanism behaves as a
> content-independent combiner rather than a content-conditional selector —
> but from the diversity of the decomposed inputs themselves. This finding
> lets us adopt a simpler architecture (a single query token, no
> orthogonality regularization) without loss of performance."

---

## Check-list avant soumission

- [x] Runs `mflickr_wavelet_type_ablation_nq1` et `voc_wavelet_type_ablation_nq1`
      terminés et intégrés (section 5.3) — pas de gagnant consistant, à
      reporter comme robustesse.
- [ ] Décision prise sur VOC 32/96-bit (best-epoch vs epoch-50), appliquée
      uniformément (section 5.1).
- [ ] Diagnostic d'attention (`measure_attention_collapse.py`) rejoué sur un
      checkpoint N=1 pour confirmer que le constat "uniforme/borné/inerte"
      tient à la configuration réellement utilisée en section 5.1, pas
      seulement à l'ancienne config N=4 (voir RESULTS.md section 9).
- [ ] Aucune mention, directe ou indirecte, du papier accepté nulle part
      dans le texte (titre, abstract, related work, method, discussion).
- [ ] Nom "MBW-DINO" absent du papier entier — remplacé par le titre
      original / une description neutre de l'architecture.
- [ ] (Interne, pas dans le papier) Texte de la section Method diffé
      manuellement contre le papier accepté — outil de similarité si
      possible — pour vérifier l'absence de reprise de phrases, avant
      soumission finale.
