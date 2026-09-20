# Effondrement des centres apprenables dans `HashLossV2` — analyse complète, mécanisme, et confrontation à l'état de l'art

**Contexte** : DINOv2 ViT-S/14 (`DINOHashBaseline`), MIRFLICKR-25K, codes 64 bits, 38 concepts, `batch_size=32`, seed 333, 50 epochs. Run source : `studies/mflickr_dino_hashlossv2_proxy_snapshot_bs32.yaml` (`HashLossV2`, `scale=15.0`, `proxy_polarization_weight=0.0`, `log_proxy_diagnostics=true`), 50 snapshots bruts des proxies (38×64), un par epoch.

---

## 1. Résumé

Les centres de classe apprenables de `HashLossV2` ne s'effondrent pas au hasard, et pas non plus par confusion sémantique. Ils s'effondrent parce que **la loss n'a pas de terme de biais, ce qui force chaque proxy à encoder le prior de sa propre classe dans sa direction** — et comme toutes les classes ont besoin du même prior négatif (87,5 % des termes du BCE sont des négatifs), elles se mettent toutes à pointer dans la même direction partagée. Plus une classe est rare, plus son prior domine sa loss, et moins il lui reste de capacité directionnelle pour encoder autre chose.

Trois chiffres résument tout :

| Mesure | epoch 1 | epoch 50 |
|---|---|---|
| Part de l'énergie des proxies portée par la **direction commune** | 3,8 % | **71,5 %** |
| **Rang effectif** du résidu spécifique aux classes (max 37) | 28,1 | **7,8** |
| Cosinus moyen entre les proxies des **10 classes les plus rares** | 0,013 | **0,946** |

Et le test qui identifie le mécanisme : la projection de chaque proxy sur la direction commune corrèle à **r = +0,897** (Spearman **+0,973**) avec le logit que le taux de base de sa classe exige, c'est-à-dire −log(p/(1−p)).

Ce phénomène porte un nom dans la littérature — **Minority Collapse** (Fang et al., PNAS 2021) — mais il n'a jamais été documenté ni dans un espace de hachage, ni en multi-label, ni avec cette décomposition. C'est précisément là que se situe la contribution possible.

---

## 2. Ce qui a été mesuré, et sur quoi

Chaque epoch, `main/engine/proxy_logger.py` sauvegarde le tenseur brut `self.proxies` (38×64 flottants, ~10 Ko). Sur les 50 snapshots, `scripts/analyze_proxy_trajectories.py` calcule :

- la matrice complète des similarités entre les 703 paires de classes, à chaque epoch (cosinus et distance euclidienne) ;
- pour chaque classe, son isolement (cosinus moyen aux 37 autres proxies) ;
- la norme de chaque proxy et son déplacement d'epoch en epoch ;
- au niveau système : rang effectif, part d'énergie de la composante commune, pas de Frobenius.

Ces mesures sont croisées avec la vérité terrain des classes, vérifiée séparément contre les fichiers d'annotation officiels MIRFLICKR (IoU = 1,000 pour les 38 colonnes, aucune ambiguïté) : fréquences par classe, NPMI et IoU pour les 703 paires.

Deux caractéristiques du jeu de données comptent pour la suite :

- **Déséquilibre R = 1651 / 20 = 82,5** entre la classe la plus fréquente (`people`) et la plus rare (`baby_r1`).
- **Parcimonie des labels** : 4,74 labels actifs par image en moyenne sur 38, donc **12,5 % de positifs et 87,5 % de négatifs** parmi tous les termes (échantillon, classe) du BCE.

---

## 3. Le phénomène, en quatre niveaux de lecture

Chaque niveau invalide ou reformule le précédent. C'est le parcours réel du diagnostic, et il vaut la peine d'être gardé dans l'article : plusieurs interprétations plausibles ont dû être abandonnées face aux mesures.

### Niveau 1 — « deux proxies fusionnent » (le symptôme d'origine)

Le diagnostic initial reposait sur un scalaire : `diag_min_proxy_hamming_distance` tombe exactement à 0 vers l'epoch 19–21. Lu seul, cela ressemble à un événement : deux classes précises se retrouvent avec le même code binaire.

C'est la lecture la plus naturelle — et c'est celle qui s'est révélée la plus trompeuse.

### Niveau 2 — ce n'est pas un événement, c'est une dérive globale

Avec les 703 paires : le cosinus moyen entre proxies passe de 0,012 (epoch 1) à 0,674 (epoch 50), en montant continûment (0,13 à l'epoch 5 ; 0,30 à 10 ; 0,51 à 20 ; 0,61 à 30) **sans saturer**. À l'epoch 50, 225 paires sur 703 (32 %) dépassent un cosinus de 0,9, et seulement 15 paires (2,1 %) sont encore en cosinus négatif.

Le pas de Frobenius de toute la matrice des proxies entre deux epochs consécutifs décroît de façon parfaitement lisse et monotone (0,547 → 0,364), **sans aucune discontinuité autour de l'epoch 19–21**.

> Le passage de `min_proxy_hamming_distance` à zéro n'est donc pas un événement. C'est le seuil de binarisation franchi au passage, au cours d'un processus continu déjà largement engagé. Le scalaire diagnostique d'origine mesurait le symptôme le plus tardif et le moins informatif du phénomène.

### Niveau 3 — la loi de taille de classe

La question suivante était : est-ce que les classes qui fusionnent sont les classes sémantiquement proches ? La réponse est non, et c'est net.

| Comparaison (703 paires, epoch 50) | Pearson | Spearman |
|---|---|---|
| cosinus final vs **NPMI** (co-occurrence corrigée du hasard) | −0,141 | −0,113 |
| cosinus final vs **IoU** (recouvrement littéral) | −0,041 (n.s.) | −0,363 |
| cosinus final vs NPMI, classes bien représentées uniquement (≥100 images, n=435) | +0,062 (n.s.) | +0,090 (n.s.) |
| cosinus final vs NPMI, **corrélation partielle** contrôlant la taille de classe | — | **+0,005** |
| cosinus final vs **min(n_i, n_j)** (taille de la plus petite des deux classes) | **−0,451** | **−0,458** |

Une fois la taille de classe prise en compte, le lien avec la similarité sémantique réelle est indiscernable de zéro. En revanche la taille, elle, prédit. Et au niveau de la classe plutôt que de la paire, la relation devient quasi déterministe :

| epoch | Spearman(nombre d'images, cosinus moyen aux 37 autres proxies) |
|---|---|
| 1 | −0,14 (aucune relation — initialisation) |
| 5 | −0,70 |
| 10 | −0,81 |
| 20 | −0,87 |
| 30 | −0,89 |
| 40 | −0,92 |
| **50** | **−0,934** (Pearson −0,940, p = 2×10⁻¹⁸) |

La loi ne s'atténue pas avec l'entraînement : elle se **renforce**. À l'epoch 50, les six proxies les mieux séparés sont exactement les six classes les plus fréquentes (`people`, `structures`, `plant_life`, `indoor`, `sky`, `people_r1`, toutes ≥ 1245 images), et l'isolement décroît monotonement jusqu'à `baby` (38 images), le proxy le plus fondu de tous.

*(Figure 4.)*

### Niveau 4 — la décomposition : la direction commune est un prior

C'est le niveau qui explique les trois précédents. On décompose chaque proxy (après `tanh`, c'est-à-dire tel qu'il entre réellement dans la loss) en une composante commune à toutes les classes et un résidu spécifique :

**P_i = μ + r_i**  où μ est la moyenne des 38 proxies et r_i ce qui reste.

| epoch | ‖μ‖ | part d'énergie de μ | cos(P_i, μ) moyen | rang effectif de {r_i} |
|---|---|---|---|---|
| 1 | 0,21 | 3,8 % | 0,195 | 28,1 |
| 10 | 0,84 | 32,5 % | 0,566 | 24,4 |
| 20 | 1,51 | 54,8 % | 0,729 | 17,6 |
| 30 | 2,09 | 64,6 % | 0,790 | 12,5 |
| 40 | 2,59 | 69,2 % | 0,818 | 9,5 |
| **50** | **3,01** | **71,5 %** | **0,832** | **7,8** |

*(Figure 1.)*

Et maintenant le test décisif. Si μ encode le prior « répondre non par défaut », alors la longueur de la projection de chaque proxy sur μ devrait suivre le logit que le taux de base de sa classe exige. C'est exactement ce qu'on observe :

**Pearson(−log(p/(1−p)), projection sur μ) = +0,897** (p = 2,7×10⁻¹⁴), **Spearman = +0,973**.

| classe | n | taux de base | logit requis | projection sur μ | cos(P, μ) |
|---|---|---|---|---|---|
| `baby_r1` | 20 | 0,50 % | −5,29 | 4,10 | **+0,974** |
| `river_r1` | 21 | 0,53 % | −5,24 | 4,08 | +0,975 |
| `baby` | 38 | 0,95 % | −4,65 | 4,02 | +0,985 |
| `sky` | 1297 | 32,4 % | −0,73 | 1,43 | +0,463 |
| `structures` | 1644 | 41,1 % | −0,36 | 0,56 | +0,215 |
| `people` | 1651 | 41,3 % | −0,35 | 0,51 | **+0,147** |

La corrélation se construit progressivement pendant l'entraînement : +0,11 (epoch 1), +0,68 (5), +0,82 (10), +0,87 (20), +0,90 (50).

*(Figure 2.)*

Autrement dit : à l'epoch 50, **le proxy de `baby_r1` est à 97,4 % la direction commune**. Il ne lui reste presque rien pour dire ce qu'est un bébé. Celui de `people` n'est qu'à 14,7 % la direction commune, et garde donc un résidu massif qui lui est propre. Corrélation entre la taille de classe et la norme du résidu spécifique : **Spearman +0,615**.

---

## 4. Le mécanisme, expliqué de bout en bout

La loss, telle qu'implémentée dans `main/losses/hash_loss.py` :

```python
h = torch.tanh(embeddings)                    # (N, 64), borné [-1, 1]
proxies_bounded = torch.tanh(self.proxies)    # (38, 64), borné [-1, 1]
logits = (h @ proxies_bounded.t()) / self.embedding_size * self.scale
bit_bce = F.binary_cross_entropy_with_logits(logits, labels.float())
```

Trois propriétés de cette formulation se combinent :

**(a) Il n'y a aucun terme de biais.** Le logit de la classe *i* est entièrement déterminé par le produit scalaire ⟨h, tanh(P_i)⟩. Il n'existe aucun paramètre libre capable d'absorber le taux de base de la classe : celui-ci doit donc être encodé dans la *direction* et la *norme* du proxy lui-même.

**(b) 87,5 % des termes du BCE sont des négatifs.** Pour une classe rare comme `baby_r1` (20 positifs contre 3980 négatifs), la quasi-totalité du gradient dit une seule chose : « produis un logit très négatif ». Le logit optimal en l'absence de toute information est −5,29.

**(c) La solution de « toujours répondre non » est une direction unique, et elle est la même pour toutes les classes.** Il suffit d'être anti-aligné avec l'embedding typique. Toutes les classes ayant besoin d'un prior négatif, toutes poussent dans cette même direction — d'où l'émergence de μ.

La conséquence en cascade :

1. μ se met à croître et finit par absorber 71,5 % de l'énergie des proxies.
2. Les classes fréquentes reçoivent assez de gradient positif pour construire un résidu propre substantiel, qui les sort de μ.
3. Les classes rares n'en reçoivent pas assez. Leur proxy reste ≈ μ. **Comme elles convergent toutes vers le même μ, elles convergent les unes vers les autres** : cosinus moyen 0,946 entre les 10 plus rares, contre 0,303 entre les 10 plus fréquentes *(Figure 3)*.
4. Le résidu, lui, perd du rang : de 28,1 dimensions effectives à 7,8 sur 37 possibles.

**Ce dernier point est le plus coûteux du point de vue du hashing, et c'est celui qu'il faut mettre en avant.** Un code de 64 bits n'a d'intérêt que s'il exploite sa capacité. À l'epoch 50, la structure inter-classes réellement encodée tient dans ~8 dimensions effectives, et une direction sur les 64 est intégralement consommée par un prior qu'un scalaire par classe aurait suffi à porter. Les mesures de `bit_balance_level0` (~39) et `worst_bit_balance_level0` (~28) observées par ailleurs sont cohérentes avec cette capacité gâchée.

### Le paradoxe des normes, résolu

Les normes brutes des proxies **croissent** de 1,11 à 3,92 sur 50 epochs, et ce sont les classes **rares** qui ont les plus grandes normes (`baby_r1` : 4,78 ; `structures` : 2,82 ; Spearman(n_images, norme) = −0,924).

Cela peut sembler contre-intuitif, et cela invalide l'explication par le `weight_decay` qui figurait dans le diagnostic précédent (voir §5). La décomposition la résout : la norme d'un proxy rare est presque entièrement celle de μ, qui grandit pour tout le monde. Les classes fréquentes ont un résidu comparable ou supérieur à leur norme totale (`structures` : ‖r‖/‖P‖ = 1,36), signe que leur résidu s'oppose partiellement à μ.

Cette croissance générale des normes est par ailleurs le comportement attendu d'une loss à produit scalaire non normalisé sur données séparables : le biais implicite de la descente de gradient fait diverger la norme des poids vers la solution à marge maximale (Soudry et al., JMLR 2018).

---

## 5. Hypothèses testées et écartées

Cette section est aussi importante que les résultats positifs : quatre explications plausibles ont été mesurées et rejetées.

**5.1 — Le terme `proxy_polarization` (rejeté).** Ablation dédiée à `scale=15` fixé, avec `proxy_polarization_weight` balayé sur [0,05 ; 0,0] : la fusion se produit dans les deux cas, et légèrement *plus tôt* sans le terme (epoch ~19 contre ~21). Aucun effet mesurable sur aucune métrique de récupération (tous les écarts ≤ 0,4 point). Conséquence : le terme a été mis à 0,0 par défaut dans les quatre configs concernées.

**5.2 — Le `weight_decay` d'AdamW sur l'optimiseur des proxies (rejeté).** L'explication retenue jusqu'ici était que le `weight_decay=1e-4` tirait les normes vers zéro, faisant dériver les proxies les uns vers les autres. **Les normes croissent** (1,11 → 3,92) : le `weight_decay` ne gagne clairement pas contre la pression de croissance. L'étude `mflickr_dino_hashlossv2_proxy_wd_ablation_bs32.yaml` déjà préparée reste utile comme confirmation empirique, mais son résultat attendu est désormais « aucun effet ».

**5.3 — La similarité sémantique entre classes (rejeté).** Voir §3, niveau 3 : corrélation partielle de +0,005 avec le NPMI une fois la taille contrôlée. Les paires apparemment sensées (`car`–`transport` à 0,92 ; `sea`–`water` à 0,92) ne se distinguent pas d'une population bien plus large de paires tout aussi fusionnées et sémantiquement sans rapport (`baby`–`river_r1` : 0,96 avec NPMI = −1,0).

**5.4 — « L'échantillon partagé tire le petit centre vers le grand » (rejeté, et inversé).** L'hypothèse était qu'une image portant plusieurs labels favoriserait le centre le mieux doté et traînerait les classes instables vers lui. Mesuré via le *containment* (fraction des images de la petite classe qui portent aussi le label de la grande) : **Spearman −0,489** ; via le nombre brut d'images partagées : **Spearman −0,568**. Le signe est négatif — plus deux classes se partagent des images, **moins** leurs proxies fusionnent.

Le contre-exemple est frappant : `baby` et `baby_r1` sont contenues à 100 % dans `people`, et pourtant leurs proxies restent quasi orthogonaux à celui de `people` (cos 0,108 et 0,073). À l'inverse, `baby`–`river_r1`, qui ne partagent aucune image, fusionnent à 0,96. Et chez les petites classes, partager ou non ne change rien : cosinus moyen 0,946 quand elles ne se croisent jamais (47 paires) contre 0,944 quand elles se croisent (31 paires).

L'interprétation qui en découle, et qui est cohérente avec le mécanisme du §4 : une co-occurrence réelle fournit au BCE un gradient informatif et cohérent pour positionner les deux proxies distinctement — le modèle doit encoder « c'est spécifiquement un bébé, pas juste des gens ». Une paire sans aucun signal de co-occurrence ne reçoit rien qui la concerne, et sa position relative est simplement laissée à la dérive commune.

**5.5 — Un point de stabilité à noter.** `diag_intra_class_variance` chute vite (0,284 → 0,213 entre les epochs 1 et 15) puis se fige (0,213–0,222 sur les 35 epochs suivants) alors que `bit_bce` continue de baisser (0,502 → 0,310). Le compromis features↔centres atteint donc son équilibre intra-classe très tôt : tout le gain ultérieur passe par le réarrangement inter-centres. Une intervention sur la dispersion intra-classe n'a plus de marge de manœuvre après l'epoch ~15.

*(Ces chiffres proviennent du run d'ablation de polarisation, bras `weight=0.0` — même configuration que le run de snapshots, mais pas le même run : ses `training_components` n'ont pas été récupérés.)*

---

## 6. Confrontation à l'état de l'art

### 6.1 Ce que la littérature a déjà nommé

**Neural Collapse.** Papyan, Han & Donoho (PNAS 2020) montrent qu'en fin d'entraînement sur données équilibrées, les moyennes de features par classe et les vecteurs du classifieur convergent vers les sommets d'un *simplex equiangular tight frame* (ETF) — la configuration qui maximise les angles entre classes. C'est le cadre de référence de tout ce qui suit.

**Minority Collapse.** Fang, He, Long & Su (PNAS 2021) étendent l'analyse aux données déséquilibrées via leur *Layer-Peeled Model*, et démontrent une transition de phase : au-delà d'un ratio de déséquilibre seuil R₀, « l'angle moyen entre classes minoritaires devient nul et tous les classifieurs minoritaires s'effondrent sur un vecteur unique ». Le réseau prédit alors des probabilités égales pour toutes les classes minoritaires, quelle que soit l'entrée.

> **C'est exactement notre observation.** Cosinus moyen de 0,946 entre les 10 classes les plus rares, à R = 82,5. Nous observons le Minority Collapse dans un espace de hachage.

Leur remède principal est le sur-échantillonnage des classes minoritaires, avec une réserve explicite : un taux trop élevé dégrade les performances de test.

**Le classifieur ETF fixe.** Yang, Chen, Li, Xie, Lin & Tao (NeurIPS 2022) posent la question frontalement — *« pourquoi dépenser un effort à apprendre un classifieur quand on connaît sa structure géométrique optimale ? »* — et montrent qu'initialiser le classifieur comme un ETF **et le geler** conduit naturellement à l'état de neural collapse *même sur données déséquilibrées*. Ils montrent aussi que la cross-entropy devient superflue et peut être remplacée par une simple perte quadratique.

**Neural Collapse multi-label.** Li, Li, Wang & Qu (ICML 2024) étendent la théorie au multi-label avec la *pick-all-label loss*. Ils démontrent une structure ETF généralisée où les features d'un échantillon multi-label valent la moyenne pondérée des vecteurs des classes qui le composent (h ∝ Σ_{k∈S} w_k). **Hypothèse critique : l'équilibre des classes de multiplicité 1.** Ils montrent que le résultat tient malgré un déséquilibre aux multiplicités supérieures, à condition que les échantillons mono-label restent équilibrés.

**Dimensional collapse.** Jing, Vincent, LeCun & Tian (ICLR 2022) documentent, en apprentissage contrastif, l'effondrement des embeddings dans un sous-espace de dimension réduite, mesuré par le spectre de valeurs singulières. Notre mesure de rang effectif (28,1 → 7,8) relève de la même famille, appliquée aux centres plutôt qu'aux features.

**Logit adjustment.** Menon, Jayasumana et al. (ICLR 2021) traitent le long-tail en ajoutant explicitement un décalage fonction du log-prior de classe aux logits, soit pendant l'entraînement, soit après coup. C'est la réponse canonique au problème identifié au §4 : ne pas laisser le prior contaminer la géométrie apprise.

**Découplage représentation/classifieur.** Kang et al. (ICLR 2020) montrent qu'en long-tail il vaut mieux apprendre la représentation puis recalibrer le classifieur séparément, notamment par normalisation des normes.

**Biais implicite et croissance des normes.** Soudry et al. (JMLR 2018) : sur données séparables, la descente de gradient fait diverger la norme des poids. Explique la croissance générale observée (1,11 → 3,92).

**Déséquilibre positif/négatif en multi-label.** Ben-Baruch, Ridnik et al. (ICCV 2021) proposent l'*Asymmetric Loss*, qui traite différemment positifs et négatifs précisément parce que les négatifs dominent massivement en multi-label. Directement applicable à nos 87,5 % de négatifs.

### 6.2 Ce que fait la littérature du hashing — et ce qu'elle évite

Le point remarquable : **presque personne, en hashing profond, ne laisse les centres complètement libres.**

**CSQ** (Yuan et al., CVPR 2020) génère les centres analytiquement à partir d'une matrice de Hadamard, garantissant une distance de Hamming de K/2 entre centres, et les **garde fixes**. Pour le multi-label, le centre d'un échantillon est le centroïde des centres de ses labels (vote binaire par bit). Les auteurs rapportent explicitement que les centres construits analytiquement surpassent les alternatives apprises.

**SHC** (Chen, Liu, Zhou, Ma, Chen & Zhang, ACM TOIS 2025) reproche à CSQ son indépendance aux données et construit des centres *sémantiques* : matrice de similarité inter-classes dérivée d'un classifieur pré-entraîné, puis optimisation sous contrainte de distance minimale (borne de Gilbert-Varshamov). Gains de 7 à 12 points de mAP. **Mono-label uniquement, aucune analyse de déséquilibre.**

**CRH / Codebook-Centric Deep Hashing** (Yin, Yin, Hou, Liu, Chen & Zhang, AAAI 2026) apprend les centres conjointement de bout en bout — mais **jamais librement** : les classes sont réassignées dynamiquement à des éléments distincts d'un codebook binaire pré-échantillonné, avec appariement un-à-un. Les auteurs sont explicites : c'est la contrainte de codebook elle-même qui empêche l'effondrement. Évalué en mono-label et en multi-label (MS COCO, NUS-WIDE).

**LTHNet** (Chen, Hou, Leng, Zhang, Lin & Zhang, SIGIR 2021) attaque le hashing long-tail avec une banque mémoire de prototypes de classe et une cross-entropy pondérée par classe. **Mono-label.**

**SPH** (*Semantic-Enhanced Proxy-Guided Hashing for Long-Tailed Image Retrieval*, IEEE TMM 2024) combine proxies, long-tail et hashing — le contexte le plus proche du nôtre. Le détail de la méthode n'a pas pu être vérifié (accès restreint) ; à traiter comme travail connexe à lire avant soumission.

### 6.3 Les manques — où se situe la contribution

Six trous identifiés, du plus solide au plus spéculatif :

**(1) Le Minority Collapse n'a jamais été mesuré dans un espace de hachage.** Fang et al. travaillent en classification mono-label sur des logits continus. Ici le collapse se produit sur des centres qui sont *binarisés* — et la binarisation crée un seuil discret (`min_proxy_hamming_distance` = 0) qui masque le processus continu sous-jacent. C'est une observation nouvelle et directement pertinente pour la communauté hashing.

**(2) La théorie multi-label existante suppose explicitement ce que notre cas viole.** Li et al. (ICML 2024) exigent l'équilibre des classes de multiplicité 1. MIRFLICKR a R = 82,5. Nous sommes précisément dans le régime que la théorie ne couvre pas, avec une observation empirique complète (50 epochs, trajectoire entière) à y opposer.

**(3) La décomposition prior/résidu, et sa validation quantitative, semblent inédites.** Aucun des travaux consultés ne décompose les centres en composante de prior partagée plus résidu spécifique, ni ne corrèle la première au log-odds du taux de base. C'est un *diagnostic mesurable* — applicable à n'importe quelle méthode à centres apprenables — et pas seulement le constat qu'un effondrement a lieu.

**(4) La littérature du hashing contourne le problème sans le diagnostiquer.** CSQ fige les centres, CRH les contraint à un codebook, SHC les pré-calcule. Tous *évitent* les centres libres ; aucun ne montre **pourquoi** les centres libres échouent, ni à quelle vitesse, ni quelles classes sont touchées en premier. C'est le chaînon manquant que ces mesures fournissent — et il justifie rétroactivement les choix de conception de tout ce sous-domaine.

**(5) Le hashing long-tail existant est mono-label.** LTHNet et SHC ne traitent pas le multi-label. Or le multi-label change la nature du problème : pas de softmax mais un BCE par bit, un prior par classe plutôt qu'une normalisation globale, 87,5 % de négatifs. Le mécanisme identifié au §4 est **spécifique au multi-label** : en mono-label avec softmax, la normalisation absorbe une partie du prior.

**(6) L'angle « capacité du code » est propre au hashing et paraît inexploité.** Le rang effectif se lit directement comme le nombre de bits qui travaillent réellement. Formuler l'effondrement comme une *perte de capacité mesurable du code* (~8 dimensions effectives utiles sur 37 possibles, plus une direction entière consommée par le prior) donne une métrique d'évaluation nouvelle, et explique pourquoi allonger les codes cesse d'aider au-delà d'un certain point. La littérature neural collapse n'a aucune raison de s'y intéresser ; la littérature hashing ne dispose pas de l'outillage pour le mesurer.

---

## 7. Les pistes, par ordre de priorité

Chaque piste est jugée sur : le mécanisme qu'elle vise, son coût, et surtout **la prédiction falsifiable** qu'elle produit. Les métriques de contrôle sont les mêmes partout : part d'énergie de μ, rang effectif du résidu, cosinus moyen des 10 classes les plus rares, plus `maphashing_level0` (référence actuelle : 83,14).

### P0 — Un biais par classe dans les logits *(le plus direct, une ligne de code)*

**Mécanisme visé** : supprimer la raison même pour laquelle le prior doit occuper une direction.

```python
self.bias = nn.Parameter(torch.zeros(num_classes))
logits = (h @ proxies_bounded.t()) / self.embedding_size * self.scale + self.bias
```

Variante plus forte, inspirée de Menon et al. : initialiser (ou figer) `bias` à `log(p_c / (1 - p_c))` calculé sur le split d'entraînement — l'information est déjà disponible dans `mflickr_class_frequencies_verified.csv`.

**Prédiction falsifiable** : la part d'énergie de μ cesse de croître et reste faible ; le rang effectif du résidu reste nettement au-dessus de 7,8 ; le cosinus moyen des 10 classes les plus rares chute franchement sous 0,946. Si μ continue de croître comme avant, l'hypothèse du prior est fausse et il faut revenir au §4.

**Pourquoi en premier** : c'est la seule piste qui attaque la cause identifiée plutôt qu'un symptôme, elle est triviale à implémenter, et elle est décisive dans les deux sens.

### P1 — Centres figés ou contraints *(déjà implémenté, à lancer)*

`HashLossV3` avec `freeze_centers=True` existe déjà dans le code. C'est la réponse de la littérature (CSQ, ETF fixe de Yang et al., codebook de CRH) et un test immédiat.

Trois variantes à comparer : centres Hadamard fixes (reproduction fidèle de CSQ), centres NPMI+MDS figés (la proposition V3, maintenant alimentée par les statistiques vérifiées), centres NPMI+MDS apprenables (V3 par défaut).

**Prédiction** : les centres figés ne peuvent pas s'effondrer par construction — l'intérêt du test n'est donc pas de savoir *si* le collapse disparaît, mais **ce que coûte ou rapporte la perte d'adaptabilité** en `maphashing_level0`. C'est la comparaison qui justifie (ou non) toute la démarche des centres apprenables.

**Point d'attention** : avec un biais par classe (P0), les centres apprenables pourraient redevenir compétitifs. P0 × P1 est un plan factoriel 2×2 naturel, et c'est probablement la table centrale de l'article.

### P2 — Régularisation de rang / orthogonalité sur la matrice de Gram

Pénaliser les termes hors-diagonale de `tanh(P) @ tanh(P).T`, ou viser explicitement une structure équiangulaire.

**Mécanisme visé** : l'effondrement de rang, directement. À distinguer d'un terme de répulsion par paires, qui réagit paire par paire après coup là où une contrainte sur le spectre agit sur la structure globale.

**Réserve** : cela combat le symptôme et non la cause. Si P0 fonctionne, P2 devient un raffinement plutôt qu'un correctif — et il risque de lutter contre les rapprochements légitimes (`sky`–`structures` monte de 0,19 à 0,65, ce qui correspond à une vraie co-occurrence).

### P3 — Rééquilibrage : sur-échantillonnage ou pondération asymétrique

Deux variantes distinctes :
- **Sampling équilibré par classe** — le remède explicitement proposé par Fang et al., avec leur réserve sur les taux trop élevés. Le code dispose déjà d'un module `main.samplers` branché via `dataset.sampler` (à inspecter pour voir ce qui existe).
- **Asymmetric Loss** (Ben-Baruch et al., ICCV 2021) — pondération différenciée positifs/négatifs, conçue exactement pour le régime à 87,5 % de négatifs.

**Prédiction** : atténuation du phénomène sans le supprimer, puisque le prior reste structurellement sans endroit où se loger.

### P4 — Découplage en deux temps

Entraîner le backbone, puis recalibrer les centres séparément (Kang et al.). Cohérent avec l'observation 5.5 : la variance intra-classe se fige dès l'epoch ~15, ce qui suggère que les deux phases pourraient être séparées sans perte.

**Coût** : modification non triviale de la boucle d'entraînement. À garder en réserve.

### P5 — Ce qu'il ne sert plus à rien de tester en priorité

L'étude `mflickr_dino_hashlossv2_proxy_wd_ablation_bs32.yaml` (`weight_decay` à 0) reste préparée et peu coûteuse, mais son hypothèse motrice est falsifiée par les normes croissantes. À lancer pour fermer proprement la question, pas pour en attendre un résultat.

---

## 8. Protocole de validation

Pour que les comparaisons soient lisibles, chaque run doit produire les mêmes mesures :

1. `log_proxy_diagnostics=true` partout, pour obtenir les snapshots par epoch.
2. `scripts/analyze_proxy_trajectories.py` pour les quatre tables (paires, isolement, normes/vitesses, système).
3. `scripts/plot_proxy_collapse_figures.py` pour les quatre figures.
4. Métriques de récupération à chaque intervalle d'évaluation (tous les 5 epochs, train + test, toutes les métriques), comme pour toutes les études précédentes.

Les quatre nombres de contrôle à reporter systématiquement : **part d'énergie de μ**, **rang effectif du résidu**, **cosinus moyen des 10 classes les plus rares**, **`maphashing_level0`**.

---

## 9. Limites de l'analyse actuelle

À énoncer explicitement dans l'article, et à traiter avant soumission :

- **Une seule graine, un seul run, un seul jeu de données.** Toutes les corrélations viennent du même run (seed 333). La loi de taille de classe à ρ = −0,93 est spectaculaire mais repose sur 38 points d'une seule exécution. Il faut au minimum 3 graines, et idéalement une réplication sur un second jeu multi-label (MS COCO ou NUS-WIDE, qui sont aussi les jeux de CRH).
- **La preuve du mécanisme est corrélationnelle.** r = +0,897 entre le prior requis et la projection est très fort, mais c'est une corrélation. Le test causal est P0 : ajouter le biais et vérifier que μ cesse de croître.
- **Les embeddings n'ont pas été sauvegardés.** L'affirmation « μ est anti-aligné avec l'embedding typique » est déduite de la structure de la loss, pas mesurée. Un snapshot périodique de l'embedding moyen la rendrait directe — et c'est une extension naturelle de `proxy_logger.py`.
- **Le snapshot de l'epoch 1 est postérieur à une epoch d'entraînement**, pas à l'initialisation. La véritable géométrie de départ (Xavier uniform) n'est pas observée. Sauvegarder un snapshot à l'epoch 0 est trivial et utile.
- **Les conclusions sur la polarisation et la variance intra-classe proviennent d'un run distinct** (bras `weight=0.0` de l'ablation de polarisation), de configuration identique mais non rejoué avec les snapshots.
- **Le rang effectif dépend du centrage.** Les valeurs rapportées portent sur le résidu (après retrait de μ) ; sans centrage, μ compterait comme une composante dominante et masquerait la mesure. À préciser dans toute figure.

---

## 10. Références

- Papyan, Han & Donoho (2020). *Prevalence of neural collapse during the terminal phase of deep learning training*. PNAS. [arXiv:2008.08186](https://arxiv.org/abs/2008.08186)
- Fang, He, Long & Su (2021). *Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training*. PNAS. [arXiv:2101.12699](https://arxiv.org/abs/2101.12699)
- Yang, Chen, Li, Xie, Lin & Tao (2022). *Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier at the End of Deep Neural Network?* NeurIPS 2022. [arXiv:2203.09081](https://arxiv.org/abs/2203.09081)
- Li, Li, Wang & Qu (2024). *Neural Collapse in Multi-label Learning with Pick-all-label Loss*. ICML 2024. [arXiv:2310.15903](https://arxiv.org/abs/2310.15903)
- Jing, Vincent, LeCun & Tian (2022). *Understanding Dimensional Collapse in Contrastive Self-Supervised Learning*. ICLR 2022. [OpenReview](https://openreview.net/pdf?id=YevsQ05DEN7)
- Menon, Jayasumana et al. (2021). *Long-tail learning via logit adjustment*. ICLR 2021. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314)
- Kang et al. (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition*. ICLR 2020. [arXiv:1910.09217](https://arxiv.org/pdf/1910.09217)
- Soudry et al. (2018). *The Implicit Bias of Gradient Descent on Separable Data*. JMLR 19. [JMLR](https://jmlr.org/papers/v19/18-188.html)
- Ben-Baruch, Ridnik et al. (2021). *Asymmetric Loss For Multi-Label Classification*. ICCV 2021. [arXiv:2009.14119](https://arxiv.org/abs/2009.14119)
- Yuan, Wang, Zhang et al. (2020). *Central Similarity Quantization for Efficient Image and Video Retrieval*. CVPR 2020. [arXiv:1908.00347](https://arxiv.org/abs/1908.00347)
- Chen, Liu, Zhou, Ma, Chen & Zhang (2025). *Deep Hashing with Semantic Hash Centers for Image Retrieval*. ACM TOIS. [DOI](https://dl.acm.org/doi/10.1145/3749983)
- Yin, Yin, Hou, Liu, Chen & Zhang (2026). *Codebook-Centric Deep Hashing: End-to-End Joint Learning of Semantic Hash Centers and Neural Hash Function*. AAAI 2026. [arXiv:2511.12162](https://arxiv.org/html/2511.12162v1)
- Chen, Hou, Leng, Zhang, Lin & Zhang (2021). *Long-Tail Hashing*. SIGIR 2021. [PDF](https://zhouchenlin.github.io/Publications/2021-SIGIR-Hashing.pdf)
- *Semantic-Enhanced Proxy-Guided Hashing for Long-Tailed Image Retrieval*. IEEE TMM 2024. [IEEE Xplore](https://ieeexplore.ieee.org/document/10509797/) — à lire, non vérifié
- Kim, Kim & Cho (2020). *Proxy Anchor Loss for Deep Metric Learning*. CVPR 2020. [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)
- Roth, Vinyals & Akata (2022). *Non-isotropy Regularization for Proxy-based Deep Metric Learning*. [arXiv:2203.08547](https://ar5iv.arxiv.org/html/2203.08547)

---

## Annexe — fichiers produits

| Fichier | Contenu |
|---|---|
| `proxy_pair_trajectory_hashlossv2_snapshot_run.csv` | 703 paires : cosinus/distance début–fin, pic, creux, oscillation, NPMI, IoU, tailles |
| `proxy_per_class_isolation_trajectory.csv` | 38 classes × 50 epochs : cosinus moyen aux autres proxies |
| `proxy_per_class_norm_velocity.csv` | 38 classes × 50 epochs : norme, déplacement, déplacement relatif |
| `proxy_system_summary.csv` | 50 epochs : normes, rang effectif, variance des 3 premières CP, pas de Frobenius |
| `proxy_shared_vs_specific_decomposition.csv` | 38 classes × 50 epochs : projection sur μ, cosinus avec μ, norme du résidu, prior requis |
| `figures/fig1_shared_component.{png,pdf}` | Composante partagée et rang effectif au fil des epochs |
| `figures/fig2_prior_encoding.{png,pdf}` | Projection sur μ vs prior requis (r = +0,897) |
| `figures/fig3_minority_collapse.{png,pdf}` | Cosinus moyen : 10 plus rares vs 10 plus fréquentes |
| `figures/fig4_class_size_law.{png,pdf}` | Isolement vs taille de classe (ρ = −0,934) |
| `scripts/analyze_proxy_trajectories.py` | Génère les quatre premières tables depuis les snapshots |
| `scripts/plot_proxy_collapse_figures.py` | Génère les quatre figures depuis les snapshots |
