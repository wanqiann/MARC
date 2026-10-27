"""End-to-end pilot for shared/private latent completion with shallow MLP mappers."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs import get_args_parser
from models.completion import MODALITIES, SharedPrivateCompletionTrainer
from utils import (
    NativeScaler,
    get_model_cols,
    get_modality_index_map,
    load_biobank,
    load_feature_metadata,
    tensor_from_frame,
)


SCENARIOS = (
    ("host_endocrine_to_brain", "brain", ("host", "endocrine")),
    ("host_brain_to_endocrine", "endocrine", ("host", "brain")),
    ("endocrine_brain_to_host", "host", ("endocrine", "brain")),
)

PARTIAL_SCENARIOS = (
    ("host_to_brain", "brain", ("host",)),
    ("brain_to_host", "host", ("brain",)),
    ("host_to_endocrine", "endocrine", ("host",)),
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_modality_feature_indices(columns, manifest):
    indices = {modality: [] for modality in MODALITIES}
    for index, modality_name in manifest["modalities"].items():
        modality_name = str(modality_name).lower()
        if modality_name in {"reproductive", "covariate"}:
            modality = "host"
        elif modality_name == "hormone":
            modality = "endocrine"
        elif modality_name == "dmri":
            modality = "brain"
        else:
            raise ValueError(
                f"Model feature {columns[int(index)]!r} has unsupported modality {modality_name!r}"
            )
        indices[modality].append(int(index))
    return indices


def validate_modality_features(modality_feature_indices):
    missing = [modality for modality in MODALITIES if not modality_feature_indices[modality]]
    if missing:
        raise ValueError(f"The current analysis queue has no model features for: {missing}")


def participant_split(df, seed, train_fraction=0.8):
    """Split unique participant IDs while retaining every row's natural mask."""
    participant_ids = df["eid"].to_numpy() if "eid" in df.columns else np.arange(len(df))
    unique_ids = pd.unique(participant_ids)
    rng = np.random.default_rng(seed)
    shuffled_ids = unique_ids[rng.permutation(len(unique_ids))]
    n_train = int(round(train_fraction * len(shuffled_ids)))
    n_train = min(max(n_train, 1), len(shuffled_ids) - 1)
    train_ids = shuffled_ids[:n_train]
    train_id_set = set(train_ids.tolist())
    train_rows = np.asarray([i for i, value in enumerate(participant_ids) if value in train_id_set])
    val_rows = np.asarray([i for i, value in enumerate(participant_ids) if value not in train_id_set])
    if set(participant_ids[train_rows].tolist()) & set(participant_ids[val_rows].tolist()):
        raise RuntimeError("Participant leakage detected in train/validation split")
    return train_rows, val_rows, train_ids, shuffled_ids[n_train:]


def _participant_availability(mask, modality_feature_indices):
    return torch.stack(
        [mask[:, indices].any(dim=1) for indices in modality_feature_indices.values()],
        dim=1,
    )


def _pearson_by_dimension(predicted, target, eps=1e-8):
    values = []
    for dimension in range(predicted.shape[1]):
        pred = predicted[:, dimension]
        truth = target[:, dimension]
        pred_centered = pred - pred.mean()
        truth_centered = truth - truth.mean()
        denominator = torch.sqrt(pred_centered.square().sum() * truth_centered.square().sum())
        if denominator <= eps:
            values.append(np.nan)
        else:
            values.append(float((pred_centered * truth_centered).sum().item() / denominator.item()))
    return values


def _train_shared_means(trainer, train_data, train_mask):
    trainer.model.eval()
    with torch.no_grad():
        output = trainer.model(train_data.to(trainer.device), train_mask.to(trainer.device))
    means = {}
    availability = output["natural_availability_mask"]
    for index, modality in enumerate(MODALITIES):
        rows = availability[:, index]
        if not rows.any():
            raise ValueError(f"No naturally observed training participants for {modality}")
        means[modality] = output["shared_by_modality"][modality][rows].mean(dim=0).detach()
    return means


def _metric_arrays(predicted, target, baseline):
    error = predicted - target
    baseline_error = baseline - target
    model_mse_by_dimension = error.square().mean(dim=0).detach().cpu().numpy()
    baseline_mse_by_dimension = baseline_error.square().mean(dim=0).detach().cpu().numpy()
    r2_by_dimension = np.divide(
        baseline_mse_by_dimension - model_mse_by_dimension,
        baseline_mse_by_dimension,
        out=np.full_like(model_mse_by_dimension, np.nan),
        where=baseline_mse_by_dimension > 1e-12,
    )
    model_mse = float(error.square().mean().item())
    baseline_mse = float(baseline_error.square().mean().item())
    model_sse = float(error.square().sum().item())
    baseline_sse = float(baseline_error.square().sum().item())
    pearson = np.asarray(_pearson_by_dimension(predicted, target), dtype=float)
    return {
        "model_mse": model_mse,
        "baseline_mse": baseline_mse,
        "r2": 1.0 - model_sse / baseline_sse if baseline_sse > 1e-12 else np.nan,
        "normalized_mse": model_mse / baseline_mse if baseline_mse > 1e-12 else np.nan,
        "model_mse_by_dimension": model_mse_by_dimension,
        "baseline_mse_by_dimension": baseline_mse_by_dimension,
        "r2_by_dimension": r2_by_dimension,
        "pearson_by_dimension": pearson,
    }


def _metric_row(scenario, target, n, method, metrics):
    pearson = metrics["pearson_by_dimension"]
    valid_pearson = pearson[np.isfinite(pearson)]
    r2 = metrics["r2_by_dimension"]
    valid_r2 = r2[np.isfinite(r2)]
    row = {
        "scenario": scenario,
        "target_modality": target,
        "method": method,
        "n": int(n),
        "model_mse": metrics["model_mse"],
        "baseline_mse": metrics["baseline_mse"],
        "normalized_mse": metrics["normalized_mse"],
        "r2_vs_train_mean": metrics["r2"],
        "mean_pearson_r": float(np.mean(valid_pearson)) if valid_pearson.size else np.nan,
        "mean_absolute_pearson_r": float(np.mean(np.abs(valid_pearson))) if valid_pearson.size else np.nan,
        "median_absolute_pearson_r": float(np.median(np.abs(valid_pearson))) if valid_pearson.size else np.nan,
        "positive_r2_dimensions": int(np.sum(valid_r2 > 0)),
        "positive_pearson_dimensions": int(np.sum(valid_pearson > 0)),
        "shared_dim": int(len(r2)),
    }
    for i, value in enumerate(metrics["model_mse_by_dimension"]):
        row[f"model_mse_dim_{i}"] = float(value)
    for i, value in enumerate(metrics["baseline_mse_by_dimension"]):
        row[f"baseline_mse_dim_{i}"] = float(value)
    for i, value in enumerate(metrics["r2_by_dimension"]):
        row[f"r2_dim_{i}"] = float(value)
    for i, value in enumerate(pearson):
        row[f"pearson_r_dim_{i}"] = float(value)
    return row


def evaluate_validation_completion(trainer, val_data, val_mask, train_means, modality_feature_indices):
    """Evaluate the three deterministic two-source completion scenarios."""
    trainer.model.eval()
    availability = _participant_availability(val_mask, modality_feature_indices)
    rows = []
    with torch.no_grad():
        for scenario, target, sources in SCENARIOS:
            target_index = MODALITIES.index(target)
            required = [target_index, *(MODALITIES.index(source) for source in sources)]
            eligible = availability[:, required].all(dim=1)
            n = int(eligible.sum().item())
            if n == 0:
                rows.append({"scenario": scenario, "target_modality": target, "method": "mlp", "n": 0})
                continue
            scenario_data = val_data[eligible].to(trainer.device)
            scenario_mask = val_mask[eligible].to(trainer.device)
            dropout = torch.zeros(n, len(MODALITIES), dtype=torch.bool, device=trainer.device)
            dropout[:, target_index] = True
            output = trainer.model(scenario_data, scenario_mask, dropout)
            predicted = output["completion_predictions"][target]
            target_latent = output["shared_by_modality"][target].detach()
            baseline = train_means[target].to(trainer.device).expand_as(target_latent)
            rows.append(_metric_row(
                scenario,
                target,
                n,
                "mlp",
                _metric_arrays(predicted, target_latent, baseline),
            ))
    frame = pd.DataFrame(rows)
    score = float(frame["normalized_mse"].mean()) if len(frame) and frame["normalized_mse"].notna().any() else np.nan
    return frame, score


def evaluate_partial_completion(trainer, val_data, val_mask, train_means, modality_feature_indices):
    """Evaluate strict single-source natural availability scenarios."""
    trainer.model.eval()
    availability = _participant_availability(val_mask, modality_feature_indices)
    rows = []
    with torch.no_grad():
        for scenario, target, sources in PARTIAL_SCENARIOS:
            target_index = MODALITIES.index(target)
            source_indices = [MODALITIES.index(source) for source in sources]
            eligible = availability[:, target_index] & availability[:, source_indices].all(dim=1)
            excluded = [index for index in range(len(MODALITIES)) if index not in [target_index, *source_indices]]
            if excluded:
                eligible &= ~availability[:, excluded].any(dim=1)
            n = int(eligible.sum().item())
            if n == 0:
                rows.append({"scenario": scenario, "target_modality": target, "n": 0})
                continue
            scenario_data = val_data[eligible].to(trainer.device)
            scenario_mask = val_mask[eligible].to(trainer.device)
            dropout = torch.zeros(n, len(MODALITIES), dtype=torch.bool, device=trainer.device)
            dropout[:, target_index] = True
            output = trainer.model(scenario_data, scenario_mask, dropout)
            predicted = output["completion_predictions"][target]
            target_latent = output["shared_by_modality"][target].detach()
            baseline = train_means[target].to(trainer.device).expand_as(target_latent)
            row = _metric_row(
                scenario,
                target,
                n,
                "mlp",
                _metric_arrays(predicted, target_latent, baseline),
            )
            row["source_modalities"] = "+".join(sources)
            rows.append(row)
    return pd.DataFrame(rows)


def write_reliability_calibration(
    trainer, val_data, val_mask, train_means, validation_ids,
    modality_feature_indices, output_dir, seed,
):
    """Calibrate completed-view reliability on the held-out validation split."""
    availability = _participant_availability(val_mask, modality_feature_indices)
    endocrine_count = val_mask[:, modality_feature_indices["endocrine"]].sum(dim=1)
    host_available, endocrine_available, brain_available = availability.unbind(dim=1)
    scenarios = [
        ("endocrine_completion", "endocrine", ("host", "brain"),
         "endocrine_6_of_6_observed", host_available & brain_available & (endocrine_count == 6)),
        ("brain_completion", "brain", ("host", "endocrine"),
         "endocrine_1_to_6_observed", host_available & brain_available & endocrine_available),
    ]
    rows, predictions = [], {}
    trainer.model.eval()
    with torch.no_grad():
        prepared = val_data.to(trainer.device)
        observed = val_mask.to(trainer.device)
        true_output = trainer.model(prepared, observed)
        for scenario, target, sources, pattern, eligible in scenarios:
            selected = eligible.to(trainer.device)
            dropout = torch.zeros(
                (len(val_data), len(MODALITIES)), dtype=torch.bool, device=trainer.device
            )
            target_index = MODALITIES.index(target)
            dropout[selected, target_index] = True
            predicted_output = trainer.model(prepared, observed, dropout)
            truth = true_output["shared_by_modality"][target][selected]
            prediction = predicted_output["completion_predictions"][target][selected]
            metrics = _metric_arrays(prediction, truth, train_means[target].to(trainer.device))
            finite_pearson = np.abs(metrics["pearson_by_dimension"])
            finite_pearson = finite_pearson[np.isfinite(finite_pearson)]
            rows.append({
                "seed": int(seed), "scenario": scenario, "target_modality": target,
                "available_source_modalities": "+".join(sources),
                "source_pattern": pattern, "n": int(selected.sum().item()),
                "model_mse": metrics["model_mse"], "baseline_mse": metrics["baseline_mse"],
                "normalized_mse": metrics["normalized_mse"],
                "r2_vs_train_mean": metrics["r2"],
                "mean_absolute_pearson_r": float(finite_pearson.mean()) if finite_pearson.size else np.nan,
            })
            ids = np.asarray(validation_ids)[eligible.cpu().numpy()]
            predictions[f"{target}_participant_id"] = ids
            predictions[f"{target}_true_latent"] = truth.cpu().numpy()
            predictions[f"{target}_predicted_latent"] = prediction.cpu().numpy()
            if target == "brain":
                predictions["brain_source_endocrine_observed_count"] = endocrine_count[eligible].cpu().numpy()
                for sensitivity_pattern, group in (
                    ("endocrine_6_of_6_observed", eligible & (endocrine_count == 6)),
                    ("endocrine_1_to_5_observed", eligible & (endocrine_count < 6)),
                ):
                    group_device = group.to(trainer.device)
                    group_true = true_output["shared_by_modality"][target][group_device]
                    group_pred = predicted_output["completion_predictions"][target][group_device]
                    group_metrics = _metric_arrays(
                        group_pred, group_true, train_means[target].to(trainer.device)
                    )
                    group_pearson = np.abs(group_metrics["pearson_by_dimension"])
                    group_pearson = group_pearson[np.isfinite(group_pearson)]
                    rows.append({
                        "seed": int(seed), "scenario": "brain_completion_source_sensitivity",
                        "target_modality": target, "available_source_modalities": "+".join(sources),
                        "source_pattern": sensitivity_pattern, "n": int(group.sum().item()),
                        "model_mse": group_metrics["model_mse"],
                        "baseline_mse": group_metrics["baseline_mse"],
                        "normalized_mse": group_metrics["normalized_mse"],
                        "r2_vs_train_mean": group_metrics["r2"],
                        "mean_absolute_pearson_r": float(group_pearson.mean()) if group_pearson.size else np.nan,
                    })
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "reliability_calibration.csv", index=False)
    np.savez_compressed(output_dir / "reliability_calibration_predictions.npz", **predictions)


def _diagnose_prepared(trainer, data, mask):
    trainer.model.eval()
    with torch.no_grad():
        output = trainer.model(data.to(trainer.device), mask.to(trainer.device))
    return trainer.model.latent_diagnostics(output)


def availability_intersections(data_mask, modality_feature_indices):
    availability = _participant_availability(data_mask, modality_feature_indices).cpu().numpy().astype(bool)
    return {
        "host_observed": int(availability[:, 0].sum()),
        "endocrine_observed": int(availability[:, 1].sum()),
        "brain_observed": int(availability[:, 2].sum()),
        "host_and_endocrine_observed": int((availability[:, 0] & availability[:, 1]).sum()),
        "host_and_brain_observed": int((availability[:, 0] & availability[:, 2]).sum()),
        "endocrine_and_brain_observed": int((availability[:, 1] & availability[:, 2]).sum()),
        "host_and_endocrine_and_brain_observed": int(availability.all(axis=1).sum()),
    }


def run_pilot(args, raw_df, model_cols, modality_feature_indices):
    train_rows, val_rows, train_ids, val_ids = participant_split(raw_df, args.seed)
    train_frame = raw_df.iloc[train_rows].loc[:, model_cols]
    val_frame = raw_df.iloc[val_rows].loc[:, model_cols]

    trainer = SharedPrivateCompletionTrainer(args, modality_feature_indices)
    train_data, train_mask = trainer._prepare_tensor(tensor_from_frame(train_frame))
    val_data, val_mask = trainer._prepare_tensor(tensor_from_frame(val_frame))
    trainer.last_dropout_summary = trainer.dropout_supervision_summary(train_mask)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "modality_feature_indices": modality_feature_indices,
        "model_features": model_cols,
        "train_participant_count": len(train_rows),
        "validation_participant_count": len(val_rows),
        "split_seed": args.seed,
    })
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "split_indices.npz",
        seed=np.asarray(args.seed, dtype=np.int64),
        train_row_indices=train_rows,
        validation_row_indices=val_rows,
        train_participant_ids=train_ids,
        validation_participant_ids=val_ids,
    )

    optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay
    )
    scaler = NativeScaler()
    history = []
    best_score = np.inf
    best_epoch = None
    best_state = None
    best_train_stats = None
    bad_epochs = 0
    early_stopping_triggered = False

    for epoch in range(args.max_epochs):
        train_stats = trainer._train_epoch(train_data, train_mask, optimizer, scaler, epoch)
        train_means = _train_shared_means(trainer, train_data, train_mask)
        validation, validation_score = evaluate_validation_completion(
            trainer, val_data, val_mask, train_means, modality_feature_indices
        )
        row = {
            "epoch": epoch + 1,
            "total_training_loss": train_stats["total_loss"],
            "L_rec": train_stats["rec_loss"],
            "L_align": train_stats["align_loss"],
            "L_align_redundancy": train_stats["align_redundancy_loss"],
            "L_orth": train_stats["orth_loss"],
            "L_completion": train_stats["completion_loss"],
            "validation_normalized_mse": validation_score,
        }
        for modality in MODALITIES:
            row[f"L_completion_{modality}"] = train_stats["completion_losses"][modality]
            row[f"completion_targets_{modality}"] = train_stats["completion_target_counts"][modality]
        history.append(row)
        if np.isfinite(validation_score) and validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in trainer.model.state_dict().items()
            }
            best_train_stats = copy.deepcopy(train_stats)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                early_stopping_triggered = True
                break

    if best_state is None:
        raise RuntimeError("No finite validation completion metric was obtained")
    trainer.model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)

    final_train_means = _train_shared_means(trainer, train_data, train_mask)
    final_validation, _ = evaluate_validation_completion(
        trainer, val_data, val_mask, final_train_means, modality_feature_indices
    )
    partial_validation = evaluate_partial_completion(
        trainer, val_data, val_mask, final_train_means, modality_feature_indices
    )
    train_diagnostics = _diagnose_prepared(trainer, train_data, train_mask)
    validation_diagnostics = _diagnose_prepared(trainer, val_data, val_mask)
    for modality in MODALITIES:
        train_diagnostics["completion"][modality] = {
            "loss": float(best_train_stats["completion_losses"][modality]),
            "target_count": int(best_train_stats["completion_target_counts"][modality]),
        }
        validation_row = final_validation.loc[final_validation["target_modality"] == modality]
        validation_diagnostics["completion"][modality] = {
            "loss": float(validation_row["model_mse"].iloc[0]) if len(validation_row) else 0.0,
            "target_count": int(validation_row["n"].iloc[0]) if len(validation_row) else 0,
        }
    diagnostics_payload = {
        "train": train_diagnostics,
        "validation": validation_diagnostics,
        "dropout_supervision_expected": trainer.last_dropout_summary,
        "best_epoch": best_epoch,
        "early_stopping_triggered": early_stopping_triggered,
        "participant_counts": {"train": len(train_rows), "validation": len(val_rows)},
        "availability_intersections": {
            "full_cohort": availability_intersections(
                torch.isfinite(tensor_from_frame(raw_df.loc[:, model_cols])),
                modality_feature_indices,
            ),
            "train": availability_intersections(train_mask, modality_feature_indices),
            "validation": availability_intersections(val_mask, modality_feature_indices),
        },
    }
    full_data, full_mask = trainer._prepare_tensor(tensor_from_frame(raw_df.loc[:, model_cols]))
    with torch.no_grad():
        full_output = trainer.model(full_data.to(trainer.device), full_mask.to(trainer.device))
    participant_id_fallback = "eid" not in raw_df.columns
    participant_ids = (
        raw_df["eid"].to_numpy()
        if not participant_id_fallback
        else np.arange(len(raw_df), dtype=np.int64)
    )
    full_availability = full_output["natural_availability_mask"].detach().cpu().numpy().astype(bool)
    np.savez_compressed(
        output_dir / "completed_shared_views.npz",
        participant_id=participant_ids,
        host_shared=full_output["S_H_complete"].detach().cpu().numpy(),
        endocrine_shared=full_output["S_E_complete"].detach().cpu().numpy(),
        brain_shared=full_output["S_B_complete"].detach().cpu().numpy(),
        host_observed=full_availability[:, 0],
        endocrine_observed=full_availability[:, 1],
        brain_observed=full_availability[:, 2],
    )
    diagnostics_payload["completed_shared_views_full_cohort"] = {
        "n_participants": int(len(participant_ids)),
        "participant_id_source": "eid" if not participant_id_fallback else "row_index_fallback",
    }
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    final_validation.to_csv(output_dir / "validation_completion.csv", index=False)
    partial_validation.to_csv(output_dir / "partial_view_diagnostics.csv", index=False)
    write_reliability_calibration(
        trainer, val_data, val_mask, final_train_means, val_ids,
        modality_feature_indices, output_dir, args.seed,
    )
    (output_dir / "latent_diagnostics.json").write_text(
        json.dumps(diagnostics_payload, indent=2), encoding="utf-8"
    )
    torch.save({
        "model_state_dict": best_state,
        "norm_parameters": trainer.norm_parameters,
        "best_epoch": best_epoch,
        "config": config,
    }, output_dir / "best_model.pt")
    return trainer, history_df, final_validation, diagnostics_payload


def summarize_final_results(results_dir):
    """Summarize completed seed_0..seed_4 outputs without rerunning any model."""
    results_dir = Path(results_dir)
    rows = []
    for seed in range(5):
        seed_dir = results_dir / f"seed_{seed}"
        validation = pd.read_csv(seed_dir / "validation_completion.csv")
        partial = pd.read_csv(seed_dir / "partial_view_diagnostics.csv")
        diagnostics = json.loads((seed_dir / "latent_diagnostics.json").read_text(encoding="utf-8"))
        history = pd.read_csv(seed_dir / "training_history.csv")
        best_epoch = int(diagnostics["best_epoch"])
        rows.append({
            "seed": f"seed_{seed}",
            "section": "training",
            "scenario": "best_epoch",
            "modality": "",
            "metric": "best_epoch",
            "value": float(best_epoch),
        })
        for _, row in validation.iterrows():
            for metric in ("r2_vs_train_mean", "model_mse", "normalized_mse", "mean_absolute_pearson_r"):
                rows.append({
                    "seed": f"seed_{seed}",
                    "section": "standard",
                    "scenario": row["scenario"],
                    "modality": row["target_modality"],
                    "metric": metric,
                    "value": float(row[metric]),
                })
        for _, row in partial.iterrows():
            for metric in ("r2_vs_train_mean", "model_mse", "normalized_mse", "mean_absolute_pearson_r"):
                rows.append({
                    "seed": f"seed_{seed}",
                    "section": "partial",
                    "scenario": row["scenario"],
                    "modality": row["target_modality"],
                    "metric": metric,
                    "value": float(row[metric]),
                })
        for modality, item in diagnostics["train"]["modalities"].items():
            for metric in ("shared_effective_rank", "mean_absolute_off_diagonal_correlation", "maximum_absolute_off_diagonal_correlation"):
                rows.append({
                    "seed": f"seed_{seed}",
                    "section": "train_latent",
                    "scenario": "",
                    "modality": modality,
                    "metric": metric,
                    "value": float(item[metric]),
                })
        for modality, item in diagnostics["validation"]["modalities"].items():
            for metric in ("shared_effective_rank", "mean_absolute_off_diagonal_correlation", "maximum_absolute_off_diagonal_correlation"):
                rows.append({
                    "seed": f"seed_{seed}",
                    "section": "validation_latent",
                    "scenario": "",
                    "modality": modality,
                    "metric": metric,
                    "value": float(item[metric]),
                })
        for split_name in ("train", "validation"):
            for modality, item in diagnostics[split_name]["completed_shared_views"].items():
                for metric in ("natural_observed_count", "naturally_completed_count", "completion_target_count"):
                    rows.append({
                        "seed": f"seed_{seed}",
                        "section": f"{split_name}_completed_views",
                        "scenario": "",
                        "modality": modality,
                        "metric": metric,
                        "value": float(item[metric]),
                    })
        _ = history  # Ensure the required per-seed history file is present.

    seed_frame = pd.DataFrame(rows)
    summary_rows = []
    grouping = ["section", "scenario", "modality", "metric"]
    for keys, group in seed_frame.groupby(grouping, dropna=False):
        section, scenario, modality, metric = keys
        summary_rows.extend([
            {
                "seed": "mean",
                "section": section,
                "scenario": scenario,
                "modality": modality,
                "metric": metric,
                "value": float(group["value"].mean()),
            },
            {
                "seed": "sd",
                "section": section,
                "scenario": scenario,
                "modality": modality,
                "metric": metric,
                "value": float(group["value"].std(ddof=1)) if len(group) > 1 else 0.0,
            },
        ])
    output = pd.concat([seed_frame, pd.DataFrame(summary_rows)], ignore_index=True)
    output.to_csv(results_dir / "summary_metrics.csv", index=False)
    return output


def main():
    args = get_args_parser().parse_args()
    set_seed(args.seed)
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")
    raw_df = load_biobank(data_path)
    metadata = load_feature_metadata(
        args.feature_metadata_path, raw_df.columns, write_if_missing=False
    )
    model_cols = get_model_cols(raw_df.columns, metadata)
    manifest = get_modality_index_map(model_cols, {c: metadata[c] for c in model_cols})
    modality_feature_indices = build_modality_feature_indices(model_cols, manifest)
    validate_modality_features(modality_feature_indices)
    if args.run_five_seeds:
        results_dir = Path(args.output_dir)
        for seed in range(5):
            seed_args = copy.deepcopy(args)
            seed_args.seed = seed
            seed_args.output_dir = str(results_dir / f"seed_{seed}")
            set_seed(seed)
            run_pilot(seed_args, raw_df, model_cols, modality_feature_indices)
        summarize_final_results(results_dir)
        return
    set_seed(args.seed)
    trainer, history, validation, diagnostics = run_pilot(
        args, raw_df, model_cols, modality_feature_indices
    )
    print(f"Pilot complete: best_epoch={diagnostics['best_epoch']}")
    print(f"Results: {Path(args.output_dir)}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()

