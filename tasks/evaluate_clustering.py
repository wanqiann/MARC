"""Summarize clustering quality and reproducibility across random seeds."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_run(root, seed, k, cluster_seed):
    path = root / "reliability" / f"seed_{seed}" / f"K_{k}" / f"cluster_seed_{cluster_seed}"
    ids = np.load(path / "participant_id.npy", allow_pickle=False)
    labels = np.load(path / "cluster_labels.npy", allow_pickle=False)
    metrics = json.loads((path / "run_metrics.json").read_text(encoding="utf-8"))
    return ids, labels, metrics


def evaluate(root, seeds, ks, cluster_seeds, output):
    rows, stability = [], []
    for k in ks:
        partitions = []
        for seed in seeds:
            for cluster_seed in cluster_seeds:
                ids, labels, metrics = load_run(root, seed, k, cluster_seed)
                rows.append({
                    "imputation_seed": seed, "clustering_seed": cluster_seed, "K": k,
                    "N": len(ids), "silhouette_cosine": metrics["silhouette_cosine"],
                    "min_cluster_fraction": metrics["min_cluster_fraction"],
                    "cluster_sizes": json.dumps(metrics["cluster_sizes"]),
                    "brain_weight": metrics["global_adaptive_weights"]["brain"],
                    "endocrine_weight": metrics["global_adaptive_weights"]["endocrine"],
                })
                partitions.append((seed, cluster_seed, ids, labels))
        for (seed_a, c_a, ids_a, labels_a), (seed_b, c_b, ids_b, labels_b) in itertools.combinations(partitions, 2):
            if not np.array_equal(ids_a, ids_b):
                pos = {value: i for i, value in enumerate(ids_b.tolist())}
                labels_b = labels_b[[pos[value] for value in ids_a.tolist()]]
            stability.append({
                "K": k, "imputation_seed_a": seed_a, "cluster_seed_a": c_a,
                "imputation_seed_b": seed_b, "cluster_seed_b": c_b,
                "ARI": adjusted_rand_score(labels_a, labels_b),
                "NMI": normalized_mutual_info_score(labels_a, labels_b),
            })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "run_metrics.csv", index=False)
    pd.DataFrame(stability).to_csv(output / "partition_stability.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results/MARC"))
    parser.add_argument("--output", type=Path, default=Path("results/MARC/evaluation"))
    parser.add_argument("--imputation_seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--k_values", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--clustering_seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    evaluate(args.results_root, args.imputation_seeds, args.k_values, args.clustering_seeds, args.output)


if __name__ == "__main__":
    main()
