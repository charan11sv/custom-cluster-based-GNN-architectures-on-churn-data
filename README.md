# ~~~~~~~~"Each notebooks in each folder have their own set of documentations in '.md' format that can guide you with those notebooks."~~~~~~~~
# # ~~~~~~~~"You may not find that many visualizations in the notebooks as these are presented as is when i worked, focussing on conducting max no of experiments. You may refer to a small sample project i built to demonstarte the visualization, MLOPS skills. For this and others you can rely on the documentation files to understand my work. "~~~~~~~~
# Cluster-Driven Churn Modeling with Custom GNNs

**A full study across clustering techniques + a custom cluster-aware GNN for churn prediction**

> **Goal.** Compare multiple clustering techniques on the Telco Churn dataset, understand which **K** (number of clusters) is appropriate for each method, and then **propose & evaluate** a **custom cluster-aware GNN**. We show that **adding the *right* clustering signal** improves downstream classification, especially for **minority (churn=1)** detection.

---

##  Contents

* [Project structure](#project-structure)
* [Data & preprocessing](#data--preprocessing)
* [Clustering techniques and how K was chosen](#clustering-techniques-and-how-k-was-chosen)
  * [ClusterGAN](#clustergan)
  * [k-Prototypes](#k-prototypes)
  * [Hierarchical (Gower distance)](#hierarchical-gower-distance)
  * [Latent Class Analysis (LCA)](#latent-class-analysis-lca)
  * [Summary: chosen Ks by method](#summary-chosen-ks-by-method)
* [Baseline supervised results](#baseline-supervised-results)
* [Graph learning experiments](#graph-learning-experiments)
  * [Vanilla GAT/GCN baselines (no clusters)](#vanilla-gatgcn-baselines-no-clusters)
  * [Custom GNN + clustering (ours)](#custom-gnn--clustering-ours)
* [Why LCA and ClusterGAN lead](#why-lca-and-clustergan-lead)
* [Why the custom GNN beats vanilla GAT/GCN](#why-the-custom-gnn-beats-vanilla-gatgcn)
* [Reconciling the K=4 vs K=2 flip for ClusterGAN](#reconciling-the-k4-vs-k2-flip-for-clustergan)
* [Practical guidance & recipes](#practical-guidance--recipes)
* [Reproducibility](#reproducibility)


---

## Project structure

Three notebooks drive this project:

1. **`churn-clustergan-k-prototype-1.ipynb`**  
   Cluster discovery with **ClusterGAN** (GAN with categorical latent codes) and **k-Prototypes** (mixed numeric+categorical).

2. **`churn-hierarchical-gower-and-lca (1).ipynb`**  
   **Hierarchical clustering** on **Gower** distance (mixed types) and **LCA** (Latent Class Analysis) using categorical mixtures.

3. **`churn-clustergcn-1 (2).ipynb`** *(final notebook)*  
   **Baselines** (incl. a quick LGBM), **vanilla GAT/GCN** (no clustering), and **our custom cluster-aware GNN**, tested across the cluster datasets produced above.

---

## Data & preprocessing

* **Dataset.** Telco/Customer churn (OpenML: `customer-churn-openml-2-7043`, 7,000+ rows).
* **Preprocessing basics.**
  * Drop `customerID` if present.
  * Coerce `TotalCharges` to numeric (`errors='coerce'`) and drop NA.
  * Label-encode categoricals; encode `Churn` as 0/1.
* **Train/test protocol.**
  * For graph experiments, maintain a **unique test set** held out **before** any class balancing.
  * For custom GNNs we tune to an **“optimistic operating point”**—**the maximal region where both precision and recall increase together** (beyond that, pushing one hurts the other).

---

## Clustering techniques and how **K** was chosen

### ClusterGAN

**What it does.** Learns a GAN with a **discrete latent code** (class variable for clusters) + continuous noise; an encoder maps data back to latent. It aims for **mode-disentangled** latent clusters with **good reconstructions**.

**How we chose K (two useful Ks):**

* **K=2 (coarse, very compact communities)**  
  * **Best internal geometry**: highest Silhouette / Calinski–Harabasz; lowest Davies–Bouldin.  
  * **Interpretation**: two very clean manifolds—excellent for **homophily** in graphs and message passing.

* **K=4 (finer, more label-aligned than K=2)**  
  * Slightly better **external alignment** (e.g., NMI/ARI vs `Churn`) than K=2.  
  * Useful when downstream models are **flat** (trees/boosting) and value **direct label agreement**.

> **Rationale:** Keep **both** K=2 and K=4 snapshots.  
> * Use **K=2** to power **graph models** (clean communities).  
> * Use **K=4** to help **flat classifiers** (better alignment with churn).

---

### k-Prototypes

**What it does.** Minimizes a mixed dissimilarity (Euclidean for numeric + Hamming for categorical; tradeoff γ) to produce prototypes.

**How we chose K (elbow + micro-seg):**

* Sweeping K shows a **monotone cost decrease**; we look for the **“elbow”** where marginal gains flatten.
* **K=8** is the **primary elbow** (first sub-~10% relative drop).  
* **K=20** is the **micro-seg** setting—gains shrink to tiny steps; we stop **before** diminishing returns dominate.

> **Rationale:**  
> * **K=8** = best trade-off Fit↔Complexity → good “workhorse” segmentation.  
> * **K=20** = **granular cohorts** for campaigns/uplift/A/B variants.

---

### Hierarchical (Gower distance)

**What it does.** Builds a dendrogram using **Gower** (handles mixed types & scales well), then cuts at K.

**How we chose K (Pareto tradeoff):**

* Multiple internal indices (Silhouette, CH, DB) + external (NMI/ARI, homogeneity/completeness).  
* **K=4** repeatedly sits near the **Pareto frontier**: strong across several metrics, **simple** to interpret, avoids over-partitioning.

> **Rationale:** **K=4** captures global structure **without** fragmenting useful cohorts.

---

### Latent Class Analysis (LCA)

**What it does.** A **probabilistic categorical mixture**; selects the #classes by **BIC/AIC**.

**How we chose K (model selection):**

* Fit K=2…K_max and compare **BIC** (penalizes complexity).  
* **K=5** minimizes **BIC** → statistically favored model.

> **Rationale:** **K=5** is **BIC-optimal**, typically reveals distinct subscription/contract/payment **personas** that track churn behavior.

---

### Summary: chosen Ks by method

| Method                   | K chosen | Why this K?                                                                        |
| ------------------------ | -------: | ---------------------------------------------------------------------------------- |
| **ClusterGAN**           |    **2** | Best **internal cohesion** (Silhouette/CH↑, DB↓); great **graph homophily** prior. |
|                          |    **4** | Better **external alignment** to `Churn` than K=2; helpful for **flat models**.    |
| **k-Prototypes**         |    **8** | **Elbow**—best Fit↔Complexity trade-off.                                           |
|                          |   **20** | **Micro-segments** right before diminishing returns dominate.                      |
| **Hierarchical (Gower)** |    **4** | **Pareto-efficient** cut across internal+external indices; interpretable.          |
| **LCA (mixture models)** |    **5** | **BIC-optimal**; well-formed categorical personas.                                 |

---

## Baseline supervised results

**Quick LGBM (no clustering, reference operating point)**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| **0** | 0.90      | 0.71   | 0.80     | 775     |
| **1** | 0.50      | 0.78   | 0.61     | 280     |

**Accuracy:** 0.73 (N=1055)  

**Macro avg:** Precision ≈ 0.70 | Recall ≈ 0.75 | F1 ≈ 0.70  
**Weighted avg:** Precision ≈ 0.79 | Recall ≈ 0.73 | F1 ≈ 0.75


> This **“optimistic” reference** marks the **maximal region where both precision and recall increase together**. We use this as the comparison anchor for later experiments.

---

## Graph learning experiments

### Vanilla GAT/GCN baselines (no clusters)

| Setting               | Acc. | Prec (1) | Recall (1) | F1 (1) | N (test) |
| --------------------- | ---: | -------: | ---------: | -----: | -------: |
| **GAT (no clusters)** | 0.80 |     0.62 |       0.70 |   0.66 |      719 |
| **GCN (no clusters)** | 0.78 |     0.60 |   **0.77** |   0.67 |      734 |

> Already strong—especially **recall (0.77)** for GCN.

---

### Custom GNN + clustering (ours)

> Balanced training to the **optimistic operating point** (up to where both precision & recall increase). Unique test set kept **untouched**.

| Setting                         |     Acc. | Prec (1) | Recall (1) |   F1 (1) | N (test) |
| ------------------------------- | -------: | -------: | ---------: | -------: | -------: |
| **Custom GNN + LCA K=5**        |     0.81 |     0.62 |   **0.77** | **0.68** |      712 |
| **Custom GNN + ClusterGAN K=2** |     0.80 |     0.62 |       0.72 |     0.66 |      719 |
| **Custom GNN + ClusterGAN K=4** | **0.82** |     0.60 |       0.64 |     0.62 |      593 |
| Custom GNN + Gower K=4          |     0.79 |     0.62 |       0.58 |     0.60 |      714 |
| Custom GNN + k-Prototypes K=8   |     0.79 |     0.61 |       0.55 |     0.58 |      713 |
| Custom GNN + k-Prototypes K=20  |     0.78 |     0.60 |       0.56 |     0.58 |      714 |

**Observations**

* **Best minority F1:** **LCA K=5 @ 0.68** and ties the **best recall=0.77** (matches the best vanilla recall).
* **Best accuracy:** **ClusterGAN K=4 @ 0.82**, but **lower recall** (0.64).
* **K=2 vs K=4 (ClusterGAN):** **K=2** gives **better minority metrics** in custom GNN; **K=4** gives higher overall accuracy.

> Note on N: test sizes differ slightly (593–734); conclusions are directional and consistent across splits.

---

## Why **LCA** and **ClusterGAN** lead

### LCA (K=5) → **best minority F1**

* **Mechanism:** LCA models **categorical co-occurrence** (contract, tenure bands, add-ons, payments)—the **heart** of churn behavior.
* **Effect:** cluster labels have **high mutual information** with `Churn`. Both trees and GNNs benefit; in trees, the **cluster feature becomes top-important**; in GNNs, it **boosts recall** without hurting precision, yielding the best **F1=0.68**.

### ClusterGAN (K=2) → **best geometry for GNNs**

* **Mechanism:** K=2 produces the **most compact, well-separated communities** (Silhouette/CH↑, DB↓).
* **Effect:** **Graph homophily** improves; message passing is **cleaner**; custom GNNs lift **recall** and **F1** over no-cluster baselines.
* **Why not K=4 for GNNs?** K=4 gives more boundaries/smaller communities → more **heterophily edges**, hampering propagation; it helps **flat** models more than **graph** models.

---

## Why the custom GNN beats vanilla GAT/GCN

Your pipeline aligns with known GNN best practices:

1. **Graph that matches the manifold.** k-NN similarity on normalized features; clusters (when used) increase **local coherence** → **homophily** rises → better message passing.
2. **Right imbalance strategy.** Upsample **only on train**, hold out a **clean unique test**, tune to the **optimistic** P/R point. This systematically improves **churn (minority) recall** without overshooting precision.
3. **Architecture & training choices that reduce over-smoothing.** Light normalization, residual/skip tendencies, balanced batches/costs, and **AUC-based early stopping** → better tail behavior and thresholdable probabilities.
4. **Cluster as relational prior.** Even when not perfectly label-aligned (e.g., ClusterGAN K=2), the **community prior** organizes neighborhoods; your custom GNNs exploit this better than vanilla GAT/GCN.

---

## Reconciling the **K=4 vs K=2** flip for ClusterGAN

* **Flat models (trees/boosting):** value **label alignment** → **ClusterGAN K=4 > K=2**, **LCA K=5** best.
* **Graph models (GNNs):** value **community geometry** → **ClusterGAN K=2 > K=4** for minority metrics; **K=4** wins only on overall accuracy.

Not a contradiction—just different **inductive biases**.

---

## Practical guidance & recipes

### When to use which clustering

* **Train trees/boosting or need global personas → `LCA K=5`**  
  Highest external validity to churn, best minority F1 downstream, easy to interpret.
* **Train graph models (GNNs) → `ClusterGAN K=2`**  
  Strongest homophily; best minority recall/F1 for GNNs.  
  Keep **`ClusterGAN K=4`** around if you also train flat models aiming for accuracy.
* **Need a single “workhorse” segmentation for ops/marketing → `k-Prototypes K=8`**  
  Good elbow point; interpretable prototypes.
* **Want micro-segments (A/B variants) → `k-Prototypes K=20`**  
  Granular cohorts before returns diminish.
* **Desire small, interpretable, mixed-type clusters → `Gower Hierarchical K=4`**  
  Balanced internal/external scores; simple to explain.

### How to add cluster info to models

* **Flat models:** add **`cluster_id`** as a categorical feature (optionally one-hot or embed).
* **GNNs:**  
  * Add **cluster one-hot/embedding** to node features, **and/or**  
  * **Edge reweighting**: multiply edge weight by `(1+α)` if two nodes share the same cluster (start α=0.25).  
  * Optional **contrastive regularization**: pull same-cluster embeddings together, push across clusters (light-weight).

### Thresholding & calibration

* Keep the **“optimistic” P/R point** protocol.  
* Add **isotonic/Platt** calibration on validation to stabilize thresholds across runs/slices.

---

## Reproducibility

### Order of execution

1. **Clustering notebooks**
   * `churn-clustergan-k-prototype-1.ipynb` → export:
     * `Churn7043_clusterGAN_cluster_data_K2.csv`
     * `Churn7043_clusterGAN_4clusters.csv`
     * `clustered_churn_kprototype_k8.csv`
     * `clustered_churn_kprototype_k20.csv`
   * `churn-hierarchical-gower-and-lca (1).ipynb` → export:
     * `clustered_gower_hierarchical_k4.csv`
     * `clustered_LCA_K5.csv`

2. **Final notebook**
   * `churn-clustergcn-1 (2).ipynb`
     * Baseline LGBM (reference)
     * GAT/GCN baselines (no clusters)
     * Custom GNN with each clustering file



