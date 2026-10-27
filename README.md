# MARC

MARC is a two-stage framework for missing-aware multimodal representation completion and reliability-aware phenotype clustering. It uses the completed Brain and Endocrine shared representations for clustering; the Host representation contributes to completion but is not a clustering view.

## Overview

```text
multimodal features and observation masks
        -> shared/private representation completion
        -> brain_shared and endocrine_shared
        -> reliability-adjusted view fusion
        -> multi-view clustering
        -> cross-seed evaluation and consensus labels
```

The completion model encodes each modality separately, learns shared and private latent representations, and predicts missing modality-level shared representations. Its feature reconstruction and latent completion objectives follow the training implementation in `models/completion.py`. The clustering stage uses separate view encoders/decoders, reconstruction and contrastive clustering objectives, learned global view weights, and fixed participant-specific reliability.

For a participant and view, reliability is 1 for an observed view and the clipped held-out completion R² for a completed view. The learned global view weights are multiplied by these reliabilities and normalized for participant-level fusion.

## Repository structure

```text
MARC/
├── models/
│   ├── completion.py
│   ├── clustering.py
│   ├── fusion.py
│   ├── losses.py
│   └── network.py
├── tasks/
│   ├── run_completion.py
│   ├── run_clustering.py
│   ├── evaluate_clustering.py
│   └── build_consensus.py
├── data/feature_type.json
├── configs.py
├── utils.py
├── requirements.txt
└── LICENSE
```

## Installation

Use Python 3.10 or newer. From this directory, install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the hardware and CUDA version used for training.

## Data format

MARC does not distribute restricted participant data. Put an authorized input table at a local path and pass it with `--data_path`. The completion input is a CSV with one row per participant, a unique `eid` column, and the feature columns described by `data/feature_type.json`. It must include the active host, endocrine, and brain/dMRI PCA features; the model metadata marks model inputs and modality membership. Do not commit participant-level source data or model outputs to a public repository.

The clustering stage expects one `seed_{n}/` folder per completion seed, containing:

- `completed_shared_views.npz` with `participant_id`, `brain_shared`, `endocrine_shared`, `brain_observed`, and `endocrine_observed` arrays.
- `reliability_calibration.csv` with the held-out R² rows for Brain completion from `host+endocrine` and Endocrine completion from `host+brain`.

These files are produced by the completion run. The expected latent output is a completed shared representation, not a table with every raw feature imputed.

## Running representation completion

From the MARC repository root, pass the location of the authorized input CSV. The five-seed setting matches the formal analysis:

```bash
python -m tasks.run_completion \
  --data_path /path/to/authorized_features.csv \
  --feature_metadata_path data/feature_type.json \
  --output_dir results/latent_final \
  --run_five_seeds
```

Outputs are written to `results/latent_final/seed_0` through `seed_4`, including completed latent views, calibration records, configurations, and checkpoints.

## Running clustering

The default grid runs K=2 and K=3 for completion seeds 0–4, with clustering seed 0, using the formal training settings (200 reconstruction pretraining epochs, then 100 clustering epochs):

```bash
python -m tasks.run_clustering \
  --input_root results/latent_final \
  --output_root results/MARC
```

To run additional clustering seeds or K values, pass `--clustering_seeds` and `--k_values`. Outputs are organized under `results/MARC/reliability/seed_{n}/K_{k}/cluster_seed_{s}/`.

## Evaluating and building consensus phenotypes

```bash
python -m tasks.evaluate_clustering \
  --results_root results/MARC \
  --output results/MARC/evaluation

python -m tasks.build_consensus \
  --results_root results/MARC \
  --output results/MARC/consensus_labels.csv
```

Evaluation reports cosine silhouette, minimum cluster fraction, and pairwise ARI/NMI across runs. Consensus construction chooses the medoid partition by mean pairwise ARI, aligns cluster IDs with Hungarian matching, and resolves vote ties with the medoid labels.

## Citation

Citation information for the MARC study will be added upon publication. The cosine k-means++ initializer follows the algorithm of Arthur and Vassilvitskii, “k-means++: The Advantages of Careful Seeding,” SODA 2007.

Clustering uses cosine k-means with greedy k-means++ initialization, provided by scikit-learn.

## License

MARC code is released under the MIT License. This license does not grant rights to restricted input data or third-party materials.
