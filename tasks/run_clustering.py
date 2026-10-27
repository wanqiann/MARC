"""Train the reliability-aware phenotype clustering stage."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score

from models.clustering import MARCClustering, fit_clustering
from models.fusion import participant_reliability


def calibration_r2(path, target):
    frame = pd.read_csv(path)
    row = frame[
        (frame.scenario == f"{target}_completion")
        & (frame.target_modality == target)
        & (frame.available_source_modalities == ("host+endocrine" if target == "brain" else "host+brain"))
    ]
    return float(np.clip(row.r2_vs_train_mean.iloc[0], 0.0, 1.0))


def run_one(args, imputation_seed, k, clustering_seed, device):
    seed_dir = args.input_root / f"seed_{imputation_seed}"
    with np.load(seed_dir / "completed_shared_views.npz", allow_pickle=False) as archive:
        participant_id = archive["participant_id"]
        brain = archive["brain_shared"].astype(np.float32)
        endocrine = archive["endocrine_shared"].astype(np.float32)
        brain_observed = archive["brain_observed"].astype(bool)
        endocrine_observed = archive["endocrine_observed"].astype(bool)

    brain_mean = brain.mean(axis=0, dtype=np.float64)
    brain_std = brain.std(axis=0, dtype=np.float64)
    endocrine_mean = endocrine.mean(axis=0, dtype=np.float64)
    endocrine_std = endocrine.std(axis=0, dtype=np.float64)
    brain_std = np.where(brain_std > 1e-12, brain_std, 1.0)
    endocrine_std = np.where(endocrine_std > 1e-12, endocrine_std, 1.0)
    brain = ((brain.astype(np.float64) - brain_mean) / brain_std).astype(np.float32)
    endocrine = (
        (endocrine.astype(np.float64) - endocrine_mean) / endocrine_std
    ).astype(np.float32)
    rho_brain = calibration_r2(seed_dir / "reliability_calibration.csv", "brain")
    rho_endocrine = calibration_r2(seed_dir / "reliability_calibration.csv", "endocrine")
    reliability = participant_reliability(
        torch.from_numpy(brain_observed), torch.from_numpy(endocrine_observed),
        rho_brain, rho_endocrine,
    ).to(device)
    views = [torch.from_numpy(brain).to(device), torch.from_numpy(endocrine).to(device)]

    random.seed(clustering_seed)
    np.random.seed(clustering_seed)
    torch.manual_seed(clustering_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(clustering_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = MARCClustering(
        [brain.shape[1], endocrine.shape[1]], args.embedding_dim, k,
        len(participant_id), clustering_seed, device,
    ).to(device)
    fit_clustering(
        model, views, reliability, batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs, epochs=args.epochs,
        knn_k=args.knn_k, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, temperature=args.temperature,
        alpha=args.alpha, beta=args.beta,
    )
    model.eval()
    with torch.no_grad():
        _, z = model(views)
        fused, effective = model.fuse(z, reliability, return_weights=True)
        labels = model.assign(fused)
    fused_np = fused.cpu().numpy()
    labels_np = labels.cpu().numpy()
    counts = np.bincount(labels_np, minlength=k)
    weights = torch.softmax(model.view_logits, dim=0).detach().cpu().numpy()

    output = args.output_root / "reliability" / f"seed_{imputation_seed}" / f"K_{k}" / f"cluster_seed_{clustering_seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "participant_id.npy", participant_id)
    np.save(output / "cluster_labels.npy", labels_np)
    np.save(output / "consensus_embedding.npy", fused_np)
    config = {
        "imputation_seed": imputation_seed, "clustering_seed": clustering_seed,
        "n_clusters": k, "fusion_mode": "reliability",
        "input_views": ["brain_shared", "endocrine_shared"],
        "embedding_dim": args.embedding_dim, "batch_size": args.batch_size,
        "pretrain_epochs": args.pretrain_epochs, "epochs": args.epochs,
        "knn_k": args.knn_k, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "temperature": args.temperature,
        "alpha": args.alpha, "beta": args.beta,
        "rho_brain": rho_brain, "rho_endocrine": rho_endocrine,
    }
    metrics = {
        **config, "N": len(participant_id), "training_status": "success",
        "global_adaptive_weights": {"brain": float(weights[0]), "endocrine": float(weights[1])},
        "cluster_sizes": counts.tolist(),
        "min_cluster_fraction": float(counts.min() / len(labels_np)),
        "silhouette_cosine": float(silhouette_score(fused_np, labels_np, metric="cosine")),
        "mean_effective_weights": effective.mean(dim=0).cpu().tolist(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=Path("results/latent_final"))
    parser.add_argument("--output_root", type=Path, default=Path("results/MARC"))
    parser.add_argument("--imputation_seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--k_values", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--clustering_seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    for seed in args.imputation_seeds:
        for k in args.k_values:
            for cluster_seed in args.clustering_seeds:
                run_one(args, seed, k, cluster_seed, device)


if __name__ == "__main__":
    main()
