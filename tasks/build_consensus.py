"""Build participant-level consensus labels from independently fitted runs."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def build_consensus(root, seeds, ks, output):
    tables = {"participant_id": None}
    for k in ks:
        partitions = {}
        for seed in seeds:
            base = root / "reliability" / f"seed_{seed}" / f"K_{k}" / "cluster_seed_0"
            ids = np.load(base / "participant_id.npy", allow_pickle=False)
            labels = np.load(base / "cluster_labels.npy", allow_pickle=False).astype(int)
            if tables["participant_id"] is None:
                tables["participant_id"] = ids
            positions = {value: i for i, value in enumerate(ids.tolist())}
            canonical = tables["participant_id"]
            partitions[seed] = labels[[positions[value] for value in canonical.tolist()]]

        pair_ari = {pair: adjusted_rand_score(partitions[pair[0]], partitions[pair[1]])
                    for pair in itertools.combinations(seeds, 2)}
        mean_ari = {seed: np.mean([score for pair, score in pair_ari.items() if seed in pair])
                    for seed in seeds}
        reference = max(seeds, key=lambda seed: (mean_ari[seed], -seed))
        ref_labels = partitions[reference]
        aligned = {}
        for seed, labels in partitions.items():
            k_count = k
            overlap = np.zeros((k_count, k_count), dtype=int)
            np.add.at(overlap, (labels, ref_labels), 1)
            source, target = linear_sum_assignment(-overlap)
            mapping = np.empty(k_count, dtype=int)
            mapping[source] = target
            aligned[seed] = mapping[labels]

        votes = np.column_stack([aligned[seed] for seed in seeds])
        counts = np.column_stack([(votes == c).sum(axis=1) for c in range(k)])
        ties = (counts == counts.max(axis=1, keepdims=True)).sum(axis=1) > 1
        consensus = counts.argmax(axis=1)
        consensus[ties] = aligned[reference][ties]
        agreement = (votes == consensus[:, None]).sum(axis=1)
        prefix = f"k{k}_"
        tables[prefix + "consensus_label"] = consensus + 1
        tables[prefix + "membership_consistency"] = agreement / len(seeds)
        tables[prefix + "agreement_count"] = agreement
        tables[prefix + "tie_flag"] = ties
        tables[prefix + "reference_seed"] = reference
        tables[prefix + "high_confidence"] = agreement / len(seeds) >= 0.8
        for seed in seeds:
            tables[prefix + f"aligned_seed{seed}_label"] = aligned[seed] + 1

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tables).to_csv(output, index=False)
    return pd.DataFrame(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results/MARC"))
    parser.add_argument("--output", type=Path, default=Path("results/MARC/consensus_labels.csv"))
    parser.add_argument("--imputation_seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--k_values", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()
    build_consensus(args.results_root, args.imputation_seeds, args.k_values, args.output)


if __name__ == "__main__":
    main()
