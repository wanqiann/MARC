"""Utilities used by the formal shared/private latent training pipeline."""

import json
import math
import os
import re

import numpy as np
import pandas as pd
import torch


LABEL_COLS = ["pain"]
PSYCHIATRIC_COLS = [
    "Anxiety", "Bipolar", "Eating", "OCD", "Stress", "Anxiety-Disorder",
    "Bipolar-Disorder", "Depression", "Eating-Disorder",
    "Obsessive-Compulsive-Disorder",
    "Reaction-To-Severe-Stress-And-Adjustment-Disorders", "Schizophrenia",
]
EXCLUDED_OUTCOME_COLS = LABEL_COLS + PSYCHIATRIC_COLS
MODEL_IGNORED_COLS = ["HRT", "OCT"]
ID_COLS = ["eid"]
METADATA_IGNORED_COLS = MODEL_IGNORED_COLS + ID_COLS
CONDITION_COLS = ["livebirth", "age"]
CONTINUOUS_COVARIATE_TARGET_COLS = ["BMI", "income", "education"]
HORMONE_TARGET_COLS = ["fshb", "prl", "ptgds", "lhb", "tshb", "esr1"]
REQUIRED_BIOBANK_COLS = []
Z_SCORE_EPS = 1e-8


def tensor_from_frame(df):
    return torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)


def is_dmri_pca_col(col):
    return re.search(r"_pc\d+$", str(col)) is not None


def infer_feature_metadata(columns):
    metadata = {}
    excluded_set = set(EXCLUDED_OUTCOME_COLS)
    condition_set = set(CONDITION_COLS)
    covariate_set = set(CONTINUOUS_COVARIATE_TARGET_COLS)
    hormone_set = set(HORMONE_TARGET_COLS)
    for col in columns:
        if col in METADATA_IGNORED_COLS:
            continue
        if col in excluded_set:
            metadata[col] = {
                "modality": "label" if col in LABEL_COLS else "psychiatric",
                "role": "excluded", "is_model_input": False,
                "is_impute_target": False, "is_continuous": False,
                "feature_type": "binary", "num_classes": 2,
                "loss_weight": 1.0,
                "eval_group": "label" if col in LABEL_COLS else "psychiatric",
            }
        elif col in condition_set:
            is_continuous = col == "age"
            metadata[col] = {
                "modality": "reproductive" if col == "livebirth" else "covariate",
                "role": "condition", "is_model_input": True,
                "is_impute_target": False, "is_continuous": is_continuous,
                "feature_type": "continuous" if is_continuous else "ordinal",
                "loss_weight": 1.0, "eval_group": "condition",
            }
            if col == "livebirth":
                metadata[col]["num_classes"] = 5
        elif col in covariate_set:
            metadata[col] = {
                "modality": "covariate", "role": "impute_target",
                "is_model_input": True, "is_impute_target": True,
                "is_continuous": True, "feature_type": "continuous",
                "loss_weight": 1.0, "eval_group": "covariate",
            }
        elif col in hormone_set:
            metadata[col] = {
                "modality": "hormone", "role": "impute_target",
                "is_model_input": True, "is_impute_target": True,
                "is_continuous": True, "feature_type": "continuous",
                "loss_weight": 1.0, "eval_group": "hormone",
            }
        elif is_dmri_pca_col(col):
            metadata[col] = {
                "modality": "dmri", "role": "impute_target",
                "is_model_input": True, "is_impute_target": True,
                "is_continuous": True, "feature_type": "continuous",
                "loss_weight": 1.0, "eval_group": "dmri",
            }
        else:
            metadata[col] = {
                "modality": "excluded", "role": "excluded",
                "is_model_input": False, "is_impute_target": False,
                "is_continuous": False, "feature_type": "excluded",
                "loss_weight": 1.0, "eval_group": "excluded",
            }
    return metadata


def validate_feature_metadata(metadata, columns):
    missing = [col for col in columns if col not in metadata]
    if missing:
        raise ValueError(f"feature metadata is missing columns: {missing}")
    required = {
        "modality", "role", "is_model_input", "is_impute_target",
        "is_continuous", "eval_group",
    }
    for col in columns:
        missing_keys = required - set(metadata[col])
        if missing_keys:
            raise ValueError(f"feature metadata for {col} is missing keys: {sorted(missing_keys)}")


def load_feature_metadata(metadata_path, columns, write_if_missing=True):
    columns = [col for col in columns if col not in METADATA_IGNORED_COLS]
    inferred = infer_feature_metadata(columns)
    metadata_path = os.fspath(metadata_path)
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    changed = False
    for col in METADATA_IGNORED_COLS:
        if col in metadata:
            metadata.pop(col)
            changed = True
    for col in columns:
        if col not in metadata:
            metadata[col] = inferred[col]
            changed = True
    fixed_role_cols = set(
        EXCLUDED_OUTCOME_COLS + CONDITION_COLS
        + CONTINUOUS_COVARIATE_TARGET_COLS + HORMONE_TARGET_COLS
    )
    fixed_role_cols.update(col for col in columns if is_dmri_pca_col(col))
    fixed_keys = {
        "modality", "role", "is_model_input", "is_impute_target",
        "is_continuous", "feature_type", "num_classes", "eval_group",
    }
    for col in fixed_role_cols:
        if col not in columns:
            continue
        expected = inferred[col]
        current = metadata.get(col, {})
        locked = fixed_keys - {"is_model_input"} if col == "livebirth" else fixed_keys
        if any(current.get(key) != expected.get(key) for key in locked):
            metadata[col] = dict(expected)
            if col == "livebirth":
                metadata[col]["is_model_input"] = bool(
                    current.get("is_model_input", expected["is_model_input"])
                )
            changed = True
    if write_if_missing and (changed or not os.path.exists(metadata_path)):
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        ordered = {col: metadata[col] for col in columns}
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(ordered, handle, indent=2)
            handle.write("\n")
    selected = {col: metadata[col] for col in columns}
    validate_feature_metadata(selected, columns)
    return selected


def _metadata_or_infer(columns, metadata=None):
    return metadata if metadata is not None else infer_feature_metadata(columns)


def get_model_cols(columns, metadata=None):
    if metadata is None:
        missing = [col for col in REQUIRED_BIOBANK_COLS if col not in columns]
        if missing:
            raise ValueError(f"biobank.csv is missing required columns: {missing}")
    metadata = _metadata_or_infer(columns, metadata)
    return [col for col in columns if metadata.get(col, {}).get("is_model_input")]


def get_modality_index_map(columns, metadata):
    """Return the feature-index to modality mapping used by completion."""
    return {"modalities": {index: metadata[column]["modality"] for index, column in enumerate(columns)}}


def load_biobank(data_path):
    df = pd.read_csv(data_path)
    for col in df.columns:
        if col not in ID_COLS:
            df[col] = pd.to_numeric(df[col])
    return df


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / max(warmup_epochs, 1)
    else:
        denominator = max(max_epochs - warmup_epochs, 1)
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / denominator)
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = tmp_lr * param_group.get("lr_scale", 1.0)
    return tmp_lr


def get_grad_norm_(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    norm_type = float(norm_type)
    if norm_type == np.inf:
        return max(parameter.grad.detach().abs().max() for parameter in parameters)
    return torch.norm(torch.stack([
        torch.norm(parameter.grad.detach(), norm_type) for parameter in parameters
    ]), norm_type)


class NativeScaler:
    """CUDA AMP scaler with an ordinary-backward CPU fallback."""

    state_dict_key = "amp_scaler"

    def __init__(self):
        self._enabled = torch.cuda.is_available()
        try:
            self._scaler = torch.amp.GradScaler("cuda", enabled=self._enabled)
        except AttributeError:
            self._scaler = torch.cuda.amp.GradScaler(enabled=self._enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None,
                 create_graph=False, update_grad=True):
        use_amp = self._enabled and loss.device.type == "cuda"
        if use_amp:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        else:
            loss.backward(create_graph=create_graph)
        if not update_grad:
            return None
        if clip_grad is not None:
            if parameters is None:
                raise ValueError("parameters are required when clip_grad is set")
            if use_amp:
                self._scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            if use_amp:
                self._scaler.unscale_(optimizer)
            norm = get_grad_norm_(parameters)
        if use_amp:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
