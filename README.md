# Multi-Stage Movie Recommender

An interpretable two-stage recommendation pipeline built with MovieLens-20M:

1. **Candidate generation** with implicit-feedback matrix factorization using
   alternating least squares (ALS).
2. **Candidate reranking** with a PyTorch two-tower model that combines
   collaborative factors with movie-genre features.

The project evaluates ranking quality with a chronological train/validation/test
split and reports Recall@K, NDCG@K, and HitRate@K. It is intentionally designed
to make the retrieval-versus-ranking trade-off visible rather than hiding all
behavior behind a single model.

## Results

The checked-in notebook records one reproducible CPU run on a 3,000-user
subsample of MovieLens-20M:

| System | Recall@10 | NDCG@10 | HitRate@10 | Recall@200 |
| --- | ---: | ---: | ---: | ---: |
| Popularity baseline | 0.0488 | 0.0580 | 0.2794 | — |
| Implicit ALS | **0.0531** | 0.0569 | **0.3074** | **0.4735** |
| ALS + two-tower reranker | 0.0487 | 0.0549 | 0.3060 | 0.4735 |

The reranker did not outperform ALS in this lightweight configuration. That is
an important result: stage two cannot recover items that stage one fails to
retrieve, and a shallow model with limited negative sampling can add complexity
without improving ranking quality. The candidate-size ablation found the
strongest top-10 result at a 50-item pool (`Recall@10 = 0.0593`,
`NDCG@10 = 0.0634`) in the recorded run.

## Project structure

| Path | Description |
| --- | --- |
| [`recommender_pipeline.ipynb`](./recommender_pipeline.ipynb) | End-to-end experiment, metrics, ablations, and interpretation |
| [`tutorial.ipynb`](./tutorial.ipynb) | Compact learning-oriented implementation of the same architecture |
| [`concepts.md`](./concepts.md) | Notes on implicit feedback, ALS, ranking metrics, and two-tower systems |
| [`requirements.txt`](./requirements.txt) | Python dependencies |

## Dataset

The repository does **not** redistribute MovieLens-20M. The dataset license
included with the download prohibits redistribution, so download it directly
from [GroupLens](https://grouplens.org/datasets/movielens/20m/) and extract it
locally as:

```text
data/ml-20m/
├── movies.csv
└── ratings.csv
```

The notebook also contains a fallback loader for MovieLens-100K when that
dataset is available at the path documented in the notebook.

## Setup

Python 3.10+ is recommended. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch JupyterLab:

```bash
jupyter lab
```

Open [`recommender_pipeline.ipynb`](./recommender_pipeline.ipynb), select the
environment created above, and run the cells from top to bottom. The experiment
uses `SEED = 42`, a 4.0 implicit-rating threshold, chronological per-user
splits, 32 ALS factors, and a 200-item candidate pool.

## Methodology

- Ratings at least 4.0 are treated as positive implicit feedback.
- Users are subsampled reproducibly to keep the CPU experiment practical.
- Each user’s history is split chronologically to avoid future-information
  leakage.
- ALS retrieves a candidate pool, excluding items already seen in training.
- A two-tower model reranks only that pool using learned user/item embeddings.
- Evaluation is performed on held-out future interactions.

## Limitations and next steps

This is an offline research prototype, not a production recommender. The
reported experiment uses one seed, a small user subsample, few ALS iterations,
and limited negative sampling. A stronger follow-up would aggregate repeated
runs with confidence intervals, tune the reranker on validation data, compare
against a production-grade implicit-ALS implementation, and measure novelty,
coverage, calibration, and online A/B-test outcomes.

## Dataset citation

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
> History and Context. ACM Transactions on Interactive Intelligent Systems
> (TiiS) 5, 4, Article 19. DOI:
> [10.1145/2827872](https://doi.org/10.1145/2827872)
