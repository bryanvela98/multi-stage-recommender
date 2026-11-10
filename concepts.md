# Multi-Stage Recommendation: Concepts and Design Notes
## Concepts, design rationale, and experiment guide

This document refreshes the concepts you need and maps each concept to the implementation in `recommender_pipeline.ipynb`. The runnable reference for every concept
lives in `tutorial.ipynb` — the full experiment deliberately renames things, so you
must **understand and adapt**, not copy verbatim.

---

## 0. The Big Picture: Why "Multi-Stage"?

A production recommender cannot score *all* items for *all* users with an
expensive model — at MovieLens-20M scale that is 138K × 27K ≈ 3.7 billion
scores. The industry pattern (YouTube, Netflix, etc.) splits the job into stages:

| Stage | Name | Model | Job | Optimizes for |
|-------|------|-------|-----|---------------|
| **1** | Candidate generation / retrieval | Matrix Factorization (ALS) | Narrow millions → ~200 items per user | **Recall** + speed |
| **2** | Re-ranking | Two-tower neural net + side features | Order the ~200 → final top-10 | **Precision / NDCG** |

Key intuition: **Stage 1 cannot recommend what it never retrieves.** So Recall@200
is the *ceiling* on final quality. Stage 2 can only reorder what Stage 1 hands it.
This is why the full experiment reports both `Recall@200` (candidate quality) and
`NDCG@10` (final ordering quality).

---

## 0.5 How to Use the Tutorial (read this first)

`tutorial.ipynb` is a **fully runnable reference** — same pipeline, deliberately
different names. Its own recommended workflow:

1. Run all cells top-to-bottom once so you see the whole pipeline working.
2. Re-read each section and connect the equations (markdown cells) to the code.
3. Re-run with **one** hyperparameter changed (factor size, pool size, epochs) and
   watch how a metric *and* runtime move — this is exactly the intuition Q5's
   ablations and Q6's reflection are testing.

The tutorial intentionally differs so you must **adapt, not copy**: function/variable
names differ, the reporting table schema differs, and some tutorial helpers fuse steps
the full experiment keeps separate (e.g. tutorial `14ffb3d1` bundles split + matrix build;
the full experiment splits these into `time_ordered_split` + `to_csr`).

### Tutorial ↔ Full Experiment cell map

| Concept | Tutorial cell (reference) | Full Experiment implementation | Status |
|---------|---------------------------|------------------------------|--------|
| Config / hyperparameters | `acbea156` (`cfg`) | inline consts (`b64653f0`, `e5d27e0f`, `728fa2c3`) | tutorial uses smaller values |
| Load + build interactions | `86e745c1`, `2c9285ad` | `2606e87c`, `b64653f0` | given |
| Chronological split | `14ffb3d1` (`chronological_holdout`) | `50e53a32` (`time_ordered_split`) | Implemented §2 |
| CSR matrix build | `14ffb3d1` (`as_feedback_matrix`) | `50e53a32` (`to_csr`) | given |
| Metrics (Recall/Hit/NDCG) | `ddcbe83a` | `83c1f84e` (`ndcg_at_k`) | Implemented §3 |
| Popularity baseline | `67902eb9` | `2e18cb70` | given |
| Implicit ALS | `65ecd8ff` (`fit_implicit_factors`) | `5e9d0ece` (`implicit_als` item update) | Implemented §5 |
| Candidate generation | `65ecd8ff` (`make_candidate_pool`) | `5e9d0ece` (`topk_candidates_from_mf`) | given |
| Side features + concat | `16416113` | `624e5073` (`user_features`/`item_features`) | Implemented §6 |
| Two-tower model | `1f407031` (`DualEncoderRanker`) | `72aa623d` (`TwoTower`) | given |
| Negative sampling | `1f407031` (`make_train_pairs`) | `72aa623d` (`build_training_pairs`) | Implemented §7a |
| Training loop | `1f407031` (`fit_reranker`) | `728fa2c3` | Implemented §7b |
| Reranking | `1f407031` (`rerank_from_pool`) | `314ce69c` (`rerank_candidates`) | Implemented §7c |
| Comparison table | `7a1937bd` | `40d399f4` | Implemented §8 |
| Candidate-size sweep | `f5631a02` (30/80/150) | `59b31aac` (50/100/200) | Implemented §8 |

> Numeric gotcha: the tutorial uses `candidate_pool_size=150`, `als_factors=24`; the
> full experiment uses **200 candidates** and **32 factors**. Adapt the values, not just the code.

---

## 1. Implicit Feedback & the Sparse Interaction Matrix

### Concept
- **Explicit feedback** = star ratings (1–5). **Implicit feedback** = "did the user
  engage?" (binary). This full experiment converts ratings ≥ 4.0 into implicit
  positives (`label = 1`) and drops everything else. So you model *engagement*,
  not *rating prediction*.
- The interaction matrix **R ∈ {0,1}^(m×n)**: rows = users, cols = items,
  `R[u,i] = 1` if user *u* positively interacted with item *i*.
- **Sparsity**: with 3000 users × thousands of items, >99% of entries are zero.
  Storing a dense matrix wastes memory; **CSR (Compressed Sparse Row)** stores only
  non-zeros + index pointers.

### Why CSR/CSC matters (Q1.d.ii answer)
- **Memory**: dense `R` would be `m×n×4 bytes`; CSR stores ~`nnz×8 bytes`. At
  20M scale dense is gigabytes, CSR is megabytes.
- **Concrete compute benefit**: `R[u].indices` gives a user's interacted items in
  **O(nnz_u)** instead of scanning the whole row — exactly what ALS, candidate
  generation, and negative sampling rely on every iteration. CSR = fast row slices;
  CSC = fast column slices (item popularity).

### Where in the notebook
Already provided: `to_csr()`. Nothing to implement here, but **understand it** for
the written answer.

---

## 2. Time-Ordered (Chronological) Split

### Concept
Each user's history is sorted by timestamp and cut into
`train | validation | test` so that **train is strictly in the past** relative to
test. You predict the *future* from the *past*.

### Why preferable to random split (Q1.d.i answer)
- Recommendation is a **temporal forecasting** problem. A random split lets the model
  "see" future interactions during training → **data leakage** → overly optimistic
  offline metrics that won't hold in deployment.
- Time-ordered split mimics real serving: train on what happened before, evaluate on
  what came next.

### Implementation detail — cell `50e53a32`, `time_ordered_split()`
Mirror the tutorial's `chronological_holdout` (cell `14ffb3d1`). Fill the block:
```python
n_test = max(1, int(round(n * test_ratio)))
n_val  = max(1, int(round(n * val_ratio)))
split1 = n - (n_val + n_test)      # end of train
split2 = n - n_test                # end of val
if split1 <= 0:                    # guard short histories
    split1 = max(1, n - 2)
    split2 = n - 1
```
`split1`/`split2` are the only variables the surrounding code uses.

---

## 3. Ranking Evaluation Metrics

You evaluate a `dict: user → ranked item list` against held-out test items.

| Metric | Formula (per user) | What it measures |
|--------|--------------------|------------------|
| **Recall@K** | (relevant ∩ top-K) / (#relevant) | Coverage — did we find them? |
| **HitRate@K** | 1 if any relevant in top-K else 0 | Did we get *at least one* right? |
| **NDCG@K** | DCG/IDCG, DCG = Σ 1/log₂(rank+1) for hits | **Ordering** — are hits ranked high? |

- **NDCG** is the discriminating metric for Stage 2: it *rewards putting relevant
  items near the top*. Recall ignores order; NDCG does not.
- IDCG = the best possible DCG (all relevant items packed at the top), used to
  normalize NDCG into [0,1].

### Implementation detail — cell `83c1f84e`, `ndcg_at_k()`
Copy the logic from tutorial cell `ddcbe83a`:
```python
dcg = 0.0
for rank, item in enumerate(pred_items[:k], start=1):
    if item in true_items:
        dcg += 1.0 / np.log2(rank + 1)
ideal_hits = min(k, len(true_items))
idcg = np.sum([1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1)])
return dcg / idcg if idcg > 0 else np.nan
```
> Tip: convert `true_items` to a `set` once for O(1) `in` checks if you find eval slow.

---

## 4. Popularity Baseline (Q1.b)

### Concept
The simplest recommender: rank items by global frequency, recommend the most popular
ones the user hasn't already seen. **Always include a baseline** — if your fancy MF
model can't beat "just recommend blockbusters," something is wrong. Popularity is
also a surprisingly strong baseline because of the long-tail/blockbuster effect.

### Where in the notebook
`build_popularity_recs()` is **already complete** (cell `2e18cb70`). Just run it.
Note it uses `R_train.sum(axis=0)` (column sums = item popularity) and excludes seen items.

---

## 5. Stage 1 — Implicit ALS Matrix Factorization (Q2)

### Concept
Learn dense latent vectors: user factors **U ∈ R^(m×f)**, item factors
**V ∈ R^(n×f)**. Predicted score = dot product `r̂_ui = u_u · v_i`. ALS
(Alternating Least Squares) fixes one matrix and solves the other in closed form,
then alternates.

**Implicit ALS (Hu et al. 2008)** adds confidence weighting: observed positives get
confidence `c_ui = 1 + α·r_ui` (here `α=20`), so the model trusts known interactions
more than the assumed-negative zeros. The per-user closed-form solve is:

```
A = (VᵀV + reg·I) + Vᵤᵀ (Cᵤ - I) Vᵤ
b = Vᵤᵀ Cᵤ · 1
u_u = A⁻¹ b
```
where `Vᵤ` are the item factors for items user *u* touched, and `Cᵤ` their confidences.
The `VᵀV + reg·I` term is precomputed **once per iteration** (the cheap trick that
makes implicit ALS scale) and only the small per-user correction is added.

### Implementation detail — cell `5e9d0ece`, `implicit_als()` item-update block
The **user update is given**. The item update is the mirror image — swap U↔V roles.
From tutorial cell `65ecd8ff`:
```python
U_i = U[idx]                              # users who interacted with item i
c   = 1.0 + alpha * np.ones(len(idx), dtype=np.float32)
CiI = c - 1.0
A   = XTX + (U_i.T * CiI) @ U_i
b   = (U_i.T * c) @ np.ones(len(idx), dtype=np.float32)
V[i] = np.linalg.solve(A, b)
```
(`XTX = UᵀU + reg·I` is already computed just above the block.)

### Candidate generation
`topk_candidates_from_mf()` is **already complete**: computes `U @ Vᵀ`, masks seen
items with `-1e9`, uses `np.argpartition` to grab top-k efficiently. Generates the
**top-200 candidates per user** for Stage 1.

### Q2.d written answers
- **(i) Convergence**: ALS has no single "loss" exposed, so plot a **proxy loss** vs.
  iterations. Easiest: modify/wrap `implicit_als` to record, per iteration, the
  weighted reconstruction error `Σ c_ui (1 − u_u·v_i)²` on observed entries (or track
  Recall@200 on validation each iter). Show it decreasing/plateauing. *(You'll need
  to add a few lines to capture this — return a list of per-iter values.)*
- **(ii) Bottleneck**: the **per-user / per-item Python `for` loop with
  `np.linalg.solve`**. It's O(m + n) solves of f×f systems per iteration, each
  O(f³), in pure Python. This dominates because it isn't vectorized — every user
  triggers a separate small dense solve. (The dense `U @ Vᵀ` scoring is the other
  cost driver at scale.)

---

## 6. Side Information / Features (Q3)

### Concept
"Side information" = metadata beyond the interaction matrix. Here:
- **Item side features**: genre multi-hot vectors (one column per genre).
- **User side features**: average genre vector over the items they interacted with
  (a "taste profile").

These let the two-tower model generalize via *content*, not just collaborative
patterns.

### Implementation detail — cell `624e5073`
`build_genre_matrix()` and `user_profiles` are given. Fill the final block — combine
collaborative (MF) factors with side features by concatenation (tutorial cell `16416113`):
```python
user_features = np.hstack([U_mf, user_profiles]).astype(np.float32)
item_features = np.hstack([V_mf, item_genres]).astype(np.float32)
```
This makes each tower's input = `[latent factors | content features]`.

### Q3.c written answers
- **(i) Most useful features**: typically genres (clear, dense signal) + the MF
  factors themselves (collaborative signal). Run with/without and compare NDCG.
- **(ii) Cold start**: a brand-new item has *no* interactions → its MF factor is
  meaningless, but its **genre vector still exists**, so the item tower can place it
  sensibly. Same for users via their profile. Side features are the only signal in
  the cold-start regime.

---

## 7. Stage 2 — Two-Tower Neural Re-ranker (Q4)

### Concept
- **Two towers**: a user MLP `f(x_u)` and an item MLP `g(x_i)`, each producing an
  embedding. Score = dot product of L2-normalized embeddings (cosine-style):
  `s(u,i) = f(x_u) · g(x_i)`.
- **Why two towers?** Towers are independent → item embeddings can be precomputed/cached;
  only the cheap dot product is per-query. Scales to retrieval/re-ranking.
- **Training**: positives (observed interactions) + **sampled negatives** (random
  items the user didn't touch), optimized with `BCEWithLogitsLoss`. Negatives give
  the contrastive signal so the model learns to *separate* relevant from irrelevant.

### Two-tower implementation (cell `72aa623d`, `TwoTower`):

**(a) Negative sampling — `build_training_pairs()`** (cell `72aa623d`):
```python
for _ in range(neg_per_pos):
    j = int(rng.integers(0, R_train.shape[1]))
    while j in user_seen[u]:                      # reject items the user has seen
        j = int(rng.integers(0, R_train.shape[1]))
    users.append(u); items.append(j); labels.append(0.0)
```
> Note: the function uses `np.random.default_rng(seed)` as `rng` — make sure that
> name is in scope (it is, defined at top of the function).

**(b) Training loop** (cell `728fa2c3`) — mini-batch SGD (tutorial `fit_reranker`):
```python
for s in range(0, n, BATCH):
    idx = perm[s:s + BATCH]
    logits = model(u_feat[idx], i_feat[idx])
    loss = loss_fn(logits, y_t[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    total_loss += float(loss.item()) * len(idx)
```

**(c) Re-ranking — `rerank_candidates()`** (cell `314ce69c`):
```python
scores = model(u_batch, i_batch).cpu().numpy()
order  = np.argsort(-scores)
out[u] = [int(cand_arr[j]) for j in order[:topk]]
```
This scores the **200 Stage-1 candidates** with the trained model and keeps the top-10.

### Q4.e written answers
- **(i) Where reranking helps most**: NDCG@10 (ordering) — Stage 2 reorders the pool
  so genuinely relevant items rise. Recall@10 may improve modestly since the
  candidate set is fixed; the gain is in *positioning*. (Be honest: the tutorial warns
  reranking doesn't *always* win — if your NDCG doesn't improve, attribute it to weak
  features / few epochs / small pool and say so.)
- **(ii) Computational trade-off**: Stage 2 adds neural training cost + per-user
  inference over 200 candidates. But it only scores 200 items/user, not all *n* — so
  it's far cheaper than running the neural model as Stage 1. Trade latency for quality.

---

## 8. End-to-End Comparison & Ablations (Q5)

### Comparison table (cell `40d399f4`)
Build a `pd.DataFrame` with one row per system (Popularity / MF / MF+TwoTower) and
columns `Recall@10, NDCG@10, HitRate@10, Recall@200, HitRate@200, TrainTimeSec`.
You already have all these dicts: `pop_metrics`, `mf_metrics_10`, `mf_metrics_200`,
`rerank_metrics`, `mf_train_time`, `tt_train_time`. Pattern after tutorial cell `7a1937bd`.

### Ablations (cell `59b31aac`)
1. **No side features**: retrain `model_noside` using only `U_mf`/`V_mf` (no genre
   concat), rerank, record Recall@10/NDCG@10. Reuse your training-loop code.
2. **Candidate size sweep** (50/100/200): for each `k_cand`, generate candidates,
   rerank, score. Discuss **quality vs. cost**: bigger pool → higher Recall ceiling
   but more reranking compute. Pattern after tutorial cell `f5631a02`.

---

## 9. Confidence Intervals & Repeated Runs (applies to ALL quantitative results)

The full experiment **requires**: (a) average across repeated runs/splits and (b) **95% CI**
for key metrics. The starter code does single runs — you must add this.

### How
Run the pipeline over **multiple seeds** (e.g. `[42, 7, 123, 2024, 99]`), collect each
metric, then:
```python
import numpy as np
def mean_ci95(vals):
    vals = np.asarray(vals, dtype=float)
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    return m, 1.96 * se           # report  m ± 1.96·SE
```
Report as `0.123 ± 0.011`. Vary the seed in: user subsampling (`rng`), ALS init,
negative sampling, and torch. Wrap your pipeline in a function `run_once(seed)` that
returns a metrics dict, loop it, aggregate. **This is the single most common missed
requirement** — budget time for it.

---

## 10. Q6 — Reflection (deployment thinking)

- **(a) Weekly offline metric to prioritize**: **NDCG@10** (or Recall@10) — it best
  proxies what users actually see (the small top-k slate) and ordering quality.
  Justify: top-k ordering correlates most with engagement; tracking it weekly catches
  ranking regressions early.
- **(b) Online risk not captured offline**: e.g. **feedback loops / popularity bias**
  (the model keeps recommending what it already promoted), **freshness/novelty**
  (offline metrics reward re-showing known-good items, hurting discovery), **position
  bias**, or **distribution shift** between offline test and live traffic. Offline
  metrics can't measure causal user-behavior change — only an **A/B test** can.

---

## 10.5 Self-Check Quiz (from Tutorial §7, cell `0c7553ce`)

Answer these from memory before opening the full experiment — each maps to a written
answer or a design choice you must defend. (The tutorial hides hints in a
`<details>` block; the mapping below is the why.)

| Tutorial quiz question | Maps to | Answer in one line |
|------------------------|---------|--------------------|
| 1. Why split by time, not randomly? | **Q1.d.i** (§2) | Random split leaks future interactions → inflated offline metrics; time-split mimics real serving. |
| 2. What breaks if you don't remove training-seen items? | "don't lose points" (§11) | Model re-recommends already-consumed items, inflating metrics and killing real utility. |
| 3. Why can Recall@pool be high but NDCG@10 only moderate? | **Q2 / Q4 trade-off** | Stage 1 optimizes coverage (recall), not top-position ordering — that's Stage 2's job. |
| 4. Practical role of negative sampling? | **Q4 (§7)** | Negatives give the contrastive signal so the model learns to *separate* relevant from irrelevant. |
| 5. If reranking doesn't beat Stage 1, name two causes. | **Q4.e.i** | Weak side features, too few epochs/signal, or a too-small/noisy candidate pool. |

If you can answer all five cleanly, your written sections (Q1, Q2, Q4, Q6) are
essentially drafted.

### Tutorial migration pitfalls (cell `b836c764`) — the easy ways to lose marks
- Forgetting to **remove already-seen items** before recommending (breaks every metric).
- **Mixing user/item index maps** between train and test (use `user2idx`/`item2idx` consistently).
- Reporting **only top-10** metrics and omitting candidate-level `Recall@200`/`HitRate@200`.

---

## 11. Recommended Solving Order (checklist)

Work top-to-bottom; the notebook must run start→finish without manual edits.

1. **Get data**: download `ml-20m.zip`, unzip into `data/ml-20m/`. ⚠️ 20M is heavy —
   `MAX_USERS=3000` subsampling keeps runtime sane. (Fallback to ml-100k exists in
   `load_dataset()` if 20M is unavailable — note which source you used in the writeup.)
2. Fill `time_ordered_split` (§2) → run, check train/val/test nnz are sensible.
3. Fill `ndcg_at_k` (§3) → run, popularity baseline should produce non-NaN metrics.
4. Fill `implicit_als` item update (§5) → run, check Recall@200 > Recall@10.
5. Fill `user_features`/`item_features` concat (§6).
6. Fill negative sampling + training loop + rerank (§7) → loss should decrease.
7. Build comparison table + both ablations (§8).
8. **Add multi-seed runs + 95% CIs** (§9) — required for full marks.
9. Add a **runtime comparison table** (ALS time vs. tower time vs. inference) — required.
10. Write all six sets of markdown answers; fill Student Info + Reproducibility
    checklist (seeds, hardware, hyperparameters).
11. Run all cells clean, export to PDF, submit `.ipynb` + PDF to BrightSpace.

### Grading map (40 pts)
| Q | Pts | Core deliverable |
|---|-----|------------------|
| Q1 | 6 | Split + popularity baseline + sparsity/time-split answers |
| Q2 | 10 | ALS + top-200 candidates + Recall@200/HitRate@200 + convergence/bottleneck |
| Q3 | 6 | Side features + two-tower inputs + cold-start answer |
| Q4 | 10 | Two-tower train + rerank + NDCG@10 + trade-off answer |
| Q5 | 6 | 3-system comparison + 2 ablations |
| Q6 | 2 | Deployment reflection |

### Don't lose easy points
- Include **CIs** and a **runtime table** (explicit reporting requirements).
- State **seeds and hyperparameters** clearly.
- Keep **outputs in the submitted cells**.
- Never recommend already-seen items (already handled — don't break it).
- Use consistent user/item index maps between train and test.

---

## 12. Reproducibility Checklist — Filled Answers

Values below come straight from this guide (§5, §7, §9, §11). Drop into the final
notebook cell.

- **Random seed(s):** base `SEED = 42`; multi-seed runs use `[42, 7, 123, 2024, 99]`
  (§9). The seed is varied in user subsampling (`rng`), ALS init, negative sampling, and
  torch.
- **Hardware used:** CPU (no GPU required); record your machine. Runtimes go in the
  runtime comparison table (§11, step 9).
- **Key hyperparameters:**
  - *Data:* implicit threshold rating ≥ 4.0 → positive (§1); `MAX_USERS = 3000`
    subsampling (§0.5, §11); time-ordered train/val/test split (§2).
  - *ALS (Stage 1):* `factors = 32`, `alpha = 20`, L2 reg term `reg·I`; top-**200**
    candidates per user (§0.5, §5).
  - *Two-tower (Stage 2):* dot product of L2-normalized embeddings, trained with
    `BCEWithLogitsLoss` on observed positives + sampled negatives (§7).
  - *Evaluation:* Recall/HitRate/NDCG@k; 95% CIs reported as `mean ± 1.96·SE` over the
    multi-seed runs (§9).
- **Any deviations from default setup:** MovieLens-20M was unavailable, so the loader
  fell back to **MovieLens-100K** (`load_dataset()` fallback, §11) — note this in the
  writeup; all code is dataset-agnostic. ALS convergence is shown via a weighted
  reconstruction-error **proxy loss** per iteration (§5), since ALS exposes no single loss.
