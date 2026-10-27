"""Shared/private latent completion model.

The model operates on modality-level representations. Feature-level missingness
is used only to build an availability-aware input to each modality encoder; the
completion task hides whole observed modalities and predicts their shared latent.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from utils import Z_SCORE_EPS, adjust_learning_rate


MODALITIES = ("host", "endocrine", "brain")


def _zero_like(value):
    return value.sum() * 0.0


class ModalityEncoder(nn.Module):
    """Encode one modality's values and feature-observation mask."""

    def __init__(self, input_dim, hidden_dim, shared_dim, private_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(2 * self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.shared_head = nn.Linear(hidden_dim, shared_dim)
        self.private_head = nn.Linear(hidden_dim, private_dim)

    def forward(self, values, observed_mask):
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        observed_mask = observed_mask.to(dtype=values.dtype)
        hidden = self.network(torch.cat([values * observed_mask, observed_mask], dim=-1))
        return self.shared_head(hidden), self.private_head(hidden)


class ModalityDecoder(nn.Module):
    """Decode a modality representation for observed-modality reconstruction."""

    def __init__(self, shared_dim, private_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(shared_dim + private_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, shared, private):
        return self.network(torch.cat([shared, private], dim=-1))


class MaskAwareCompletionMLP(nn.Module):
    """A shallow completion mapper with explicit source availability inputs."""

    def __init__(self, input_dim, shared_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, shared_dim),
        )

    def forward(self, values):
        return self.network(values)


class SharedPrivateCompletionModel(nn.Module):
    """Shared/private representation model with deterministic shared completion."""

    def __init__(
        self,
        modality_feature_indices,
        shared_dim=8,
        private_dims=None,
        hidden_dim=64,
        lambda_rec=1.0,
        lambda_align=0.1,
        lambda_orth=0.1,
        lambda_completion=1.0,
        alignment_variance_weight=1.0,
        alignment_redundancy_weight=1.0,
        variance_floor=0.5,
    ):
        super().__init__()
        self.modalities = MODALITIES
        self.modality_feature_indices = {
            modality: [int(i) for i in modality_feature_indices.get(modality, [])]
            for modality in MODALITIES
        }
        if any(not indices for indices in self.modality_feature_indices.values()):
            missing = [m for m in MODALITIES if not self.modality_feature_indices[m]]
            raise ValueError(f"Each modality needs at least one model feature; missing {missing}")
        self.feature_to_modality = {
            index: modality
            for modality, indices in self.modality_feature_indices.items()
            for index in indices
        }
        all_indices = sorted(self.feature_to_modality)
        if len(self.feature_to_modality) != len(all_indices) or all_indices != list(range(max(all_indices) + 1)):
            raise ValueError("modality_feature_indices must cover every model feature index exactly")
        self.input_dim = max(all_indices) + 1
        self.shared_dim = int(shared_dim)
        if self.shared_dim <= 0:
            raise ValueError("shared_dim must be positive")
        private_dims = private_dims or {m: shared_dim for m in MODALITIES}
        self.private_dims = {m: int(private_dims[m]) for m in MODALITIES}
        if any(dim <= 0 for dim in self.private_dims.values()):
            raise ValueError("private dimensions must be positive")
        self.lambda_rec = float(lambda_rec)
        self.lambda_align = float(lambda_align)
        self.lambda_orth = float(lambda_orth)
        self.lambda_completion = float(lambda_completion)
        self.alignment_variance_weight = float(alignment_variance_weight)
        self.alignment_redundancy_weight = float(alignment_redundancy_weight)
        self.variance_floor = float(variance_floor)
        if (
            self.alignment_variance_weight < 0
            or self.alignment_redundancy_weight < 0
            or self.variance_floor < 0
        ):
            raise ValueError("alignment weights and variance_floor must be non-negative")

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.completion_mappers = nn.ModuleDict()
        for modality in MODALITIES:
            feature_count = len(self.modality_feature_indices[modality])
            self.encoders[modality] = ModalityEncoder(
                feature_count, hidden_dim, self.shared_dim, self.private_dims[modality]
            )
            self.decoders[modality] = ModalityDecoder(
                self.shared_dim, self.private_dims[modality], hidden_dim, feature_count
            )
            self.completion_mappers[modality] = MaskAwareCompletionMLP(
                2 * self.shared_dim + 2, self.shared_dim
            )

    def _validate_inputs(self, data, observed_mask):
        if data.dim() != 2 or data.shape[1] != self.input_dim:
            raise ValueError(f"data must have shape [B, {self.input_dim}], got {tuple(data.shape)}")
        if observed_mask is None:
            observed_mask = torch.isfinite(data)
        observed_mask = observed_mask.to(device=data.device, dtype=torch.bool)
        if observed_mask.shape != data.shape:
            raise ValueError(
                f"observed_mask must have shape {tuple(data.shape)}, got {tuple(observed_mask.shape)}"
            )
        return observed_mask

    def _encode_modalities(self, data, observed_mask):
        shared = []
        private = []
        availability = []
        for modality in MODALITIES:
            indices = self.modality_feature_indices[modality]
            values = data[:, indices]
            mask = observed_mask[:, indices]
            shared_m, private_m = self.encoders[modality](values, mask)
            shared.append(shared_m)
            # Private representations are explicitly zeroed for natural
            # missing modalities; zero is an unavailable state, not a sample.
            private.append(private_m * mask.any(dim=1, keepdim=True).to(private_m.dtype))
            availability.append(mask.any(dim=1))
        return torch.stack(shared, dim=1), tuple(private), torch.stack(availability, dim=1)

    def forward(self, data, observed_mask=None, artificial_dropout_mask=None):
        observed_mask = self._validate_inputs(data, observed_mask)
        full_shared, private_by_modality, natural_availability = self._encode_modalities(
            data, observed_mask
        )
        batch_size = data.shape[0]
        if artificial_dropout_mask is None:
            artificial_dropout_mask = torch.zeros(
                batch_size, len(MODALITIES), dtype=torch.bool, device=data.device
            )
        else:
            artificial_dropout_mask = artificial_dropout_mask.to(device=data.device, dtype=torch.bool)
            if artificial_dropout_mask.shape != (batch_size, len(MODALITIES)):
                raise ValueError(
                    "artificial_dropout_mask must have shape "
                    f"{(batch_size, len(MODALITIES))}, got {tuple(artificial_dropout_mask.shape)}"
                )
            if (artificial_dropout_mask & ~natural_availability).any():
                raise ValueError("Cannot artificially drop a naturally missing modality")

        completion_predictions = {}
        completion_source_availability = {}
        for target_index, target in enumerate(MODALITIES):
            source_indices = [index for index in range(len(MODALITIES)) if index != target_index]
            source_available = natural_availability[:, source_indices]
            source_available = source_available & ~artificial_dropout_mask[:, source_indices]
            source_tokens = full_shared[:, source_indices, :] * source_available.unsqueeze(-1).to(full_shared.dtype)
            mapper_input = torch.cat(
                [source_tokens.reshape(batch_size, -1), source_available.to(full_shared.dtype)], dim=1
            )
            completion_source_availability[target] = source_available.any(dim=1)
            completion_predictions[target] = self.completion_mappers[target](mapper_input)
        completed_shared = {}
        for index, modality in enumerate(MODALITIES):
            observed = natural_availability[:, index].unsqueeze(1)
            completed_shared[modality] = torch.where(
                observed, full_shared[:, index, :], completion_predictions[modality]
            )
        reconstructions = {
            modality: self.decoders[modality](full_shared[:, i, :], private_by_modality[i])
            for i, modality in enumerate(MODALITIES)
        }
        return {
            "shared_by_modality": {m: full_shared[:, i, :] for i, m in enumerate(MODALITIES)},
            "private_by_modality": {m: private_by_modality[i] for i, m in enumerate(MODALITIES)},
            "natural_availability_mask": natural_availability,
            "artificial_dropout_mask": artificial_dropout_mask,
            "completion_predictions": completion_predictions,
            "completion_source_availability": completion_source_availability,
            "completed_shared_by_modality": completed_shared,
            "S_H_complete": completed_shared["host"],
            "S_E_complete": completed_shared["endocrine"],
            "S_B_complete": completed_shared["brain"],
            "reconstructions": reconstructions,
            "observed_mask": observed_mask,
        }

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.to(dtype=values.dtype)
        denominator = mask.sum()
        if denominator.item() == 0:
            return _zero_like(values)
        return (values * mask).sum() / denominator

    def _reconstruction_loss(self, data, output):
        losses = []
        observed_mask = output["observed_mask"]
        for modality in MODALITIES:
            indices = self.modality_feature_indices[modality]
            prediction = output["reconstructions"][modality]
            target = torch.nan_to_num(data[:, indices], nan=0.0, posinf=0.0, neginf=0.0)
            mask = observed_mask[:, indices]
            loss = self._masked_mean((prediction - target) ** 2, mask)
            if mask.any():
                losses.append(loss)
        return torch.stack(losses).mean() if losses else _zero_like(data)

    def _alignment_components(self, output):
        shared = output["shared_by_modality"]
        available = output["natural_availability_mask"]
        correlation_losses = []
        for left in range(len(MODALITIES)):
            for right in range(left + 1, len(MODALITIES)):
                pair = available[:, left] & available[:, right]
                if pair.sum().item() < 2:
                    continue
                left_values = shared[MODALITIES[left]][pair]
                right_values = shared[MODALITIES[right]][pair]
                left_centered = left_values - left_values.mean(dim=0, keepdim=True)
                right_centered = right_values - right_values.mean(dim=0, keepdim=True)
                covariance = (left_centered * right_centered).mean(dim=0)
                left_scale = left_centered.square().mean(dim=0).sqrt().clamp_min(Z_SCORE_EPS)
                right_scale = right_centered.square().mean(dim=0).sqrt().clamp_min(Z_SCORE_EPS)
                correlation = covariance / (left_scale * right_scale)
                # CCA-inspired: maximize correlation magnitude, rather than
                # forcing shared coordinates from different modalities equal.
                correlation_losses.append((1.0 - correlation.square().clamp(max=1.0)).mean())

        # Anti-collapse component: preserve participant-level variation in
        # each modality's shared representation.  Each modality contributes
        # one mean penalty, so high-availability modalities cannot dominate.
        variance_losses = []
        for index, modality in enumerate(MODALITIES):
            modality_rows = available[:, index]
            if modality_rows.sum().item() < 2:
                continue
            values = shared[modality][modality_rows]
            std = values.std(dim=0, unbiased=False)
            variance_losses.append(torch.relu(self.variance_floor - std).mean())

        redundancy_losses = []
        for index, modality in enumerate(MODALITIES):
            modality_rows = available[:, index]
            if modality_rows.sum().item() < 2:
                continue
            values = shared[modality][modality_rows]
            centered = values - values.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, unbiased=False)
            standardized = centered / (std + Z_SCORE_EPS)
            correlation_matrix = standardized.transpose(0, 1) @ standardized
            correlation_matrix = correlation_matrix / max(values.shape[0] - 1, 1)
            off_diagonal = ~torch.eye(
                correlation_matrix.shape[0], dtype=torch.bool, device=correlation_matrix.device
            )
            redundancy_losses.append(correlation_matrix[off_diagonal].square().mean())

        correlation_loss = (
            torch.stack(correlation_losses).mean()
            if correlation_losses
            else _zero_like(next(iter(shared.values())))
        )
        variance_loss = (
            torch.stack(variance_losses).mean()
            if variance_losses
            else _zero_like(next(iter(shared.values())))
        )
        redundancy_loss = (
            torch.stack(redundancy_losses).mean()
            if redundancy_losses
            else _zero_like(next(iter(shared.values())))
        )
        return correlation_loss, variance_loss, redundancy_loss

    def _alignment_loss(self, output):
        correlation_loss, variance_loss, redundancy_loss = self._alignment_components(output)
        return (
            correlation_loss
            + self.alignment_variance_weight * variance_loss
            + self.alignment_redundancy_weight * redundancy_loss
        )

    def _orthogonality_loss(self, output):
        losses = []
        available = output["natural_availability_mask"]
        for index, modality in enumerate(MODALITIES):
            pair = available[:, index]
            if pair.sum().item() < 2:
                continue
            shared = output["shared_by_modality"][modality][pair]
            private = output["private_by_modality"][modality][pair]
            shared = shared - shared.mean(dim=0, keepdim=True)
            private = private - private.mean(dim=0, keepdim=True)
            cross_covariance = shared.transpose(0, 1) @ private / max(shared.shape[0] - 1, 1)
            losses.append(cross_covariance.square().mean())
        return torch.stack(losses).mean() if losses else _zero_like(
            next(iter(output["shared_by_modality"].values()))
        )

    def _completion_losses(self, output):
        losses = {}
        counts = {}
        valid_targets = output["artificial_dropout_mask"] & output["natural_availability_mask"]
        for index, modality in enumerate(MODALITIES):
            mask = valid_targets[:, index] & output["completion_source_availability"][modality]
            counts[modality] = mask.sum()
            if not mask.any():
                losses[modality] = _zero_like(
                    next(iter(output["shared_by_modality"].values()))
                )
                continue
            predicted = output["completion_predictions"][modality][mask]
            target = output["shared_by_modality"][modality].detach()[mask]
            losses[modality] = (predicted - target).square().mean()
        active_losses = [losses[modality] for modality in MODALITIES if counts[modality].item() > 0]
        total = torch.stack(active_losses).mean() if active_losses else _zero_like(
            next(iter(output["shared_by_modality"].values()))
        )
        return total, losses, counts

    @staticmethod
    def _shared_redundancy_diagnostics(values):
        dimension = values.shape[1]
        if values.shape[0] < 2:
            return {
                "mean_absolute_off_diagonal_correlation": 0.0,
                "maximum_absolute_off_diagonal_correlation": 0.0,
                "correlation_eigenvalues": [0.0] * dimension,
                "normalized_correlation_eigenvalues": [0.0] * dimension,
            }
        centered = values - values.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False)
        standardized = centered / (std + Z_SCORE_EPS)
        correlation = standardized.transpose(0, 1) @ standardized
        correlation = correlation / max(values.shape[0] - 1, 1)
        off_diagonal = ~torch.eye(dimension, dtype=torch.bool, device=values.device)
        off_values = correlation[off_diagonal]
        symmetric_correlation = (correlation + correlation.transpose(0, 1)) / 2.0
        eigenvalues = torch.linalg.eigvalsh(symmetric_correlation).clamp_min(0.0)
        eigen_sum = eigenvalues.sum()
        normalized = eigenvalues / eigen_sum.clamp_min(Z_SCORE_EPS)
        return {
            "mean_absolute_off_diagonal_correlation": float(off_values.abs().mean().item())
            if off_values.numel() else 0.0,
            "maximum_absolute_off_diagonal_correlation": float(off_values.abs().max().item())
            if off_values.numel() else 0.0,
            "correlation_eigenvalues": eigenvalues.detach().cpu().tolist(),
            "normalized_correlation_eigenvalues": normalized.detach().cpu().tolist(),
        }

    def latent_diagnostics(self, output):
        """Return compact, batch-level representation diagnostics."""
        diagnostics = {"modalities": {}, "completed_shared_views": {}}
        available = output["natural_availability_mask"]

        def stats(values):
            if values.shape[0] == 0:
                zeros = [0.0] * values.shape[1]
                return {"mean": zeros, "std": zeros, "effective_rank": 0.0}
            centered = values - values.mean(dim=0, keepdim=True)
            std = values.std(dim=0, unbiased=False)
            if values.shape[0] < 2 or torch.all(centered == 0):
                effective_rank = 0.0
            else:
                singular_values = torch.linalg.svdvals(centered)
                singular_sum = singular_values.sum()
                effective_rank = float(
                    ((singular_sum.square() / singular_values.square().sum().clamp_min(Z_SCORE_EPS)).item())
                )
            return {
                "mean": values.mean(dim=0).detach().cpu().tolist(),
                "std": std.detach().cpu().tolist(),
                "effective_rank": effective_rank,
            }

        for index, modality in enumerate(MODALITIES):
            rows = available[:, index]
            shared = output["shared_by_modality"][modality][rows]
            private = output["private_by_modality"][modality][rows]
            shared_stats = stats(shared)
            private_stats = stats(private)
            redundancy_stats = self._shared_redundancy_diagnostics(shared)
            if shared.shape[0] < 2:
                cross_covariance_norm = 0.0
            else:
                shared_centered = shared - shared.mean(dim=0, keepdim=True)
                private_centered = private - private.mean(dim=0, keepdim=True)
                cross_covariance = shared_centered.transpose(0, 1) @ private_centered
                cross_covariance = cross_covariance / max(shared.shape[0] - 1, 1)
                cross_covariance_norm = float(torch.linalg.matrix_norm(cross_covariance).item())
            diagnostics["modalities"][modality] = {
                "n_observed": int(rows.sum().item()),
                "shared_mean": shared_stats["mean"],
                "shared_std": shared_stats["std"],
                "private_mean": private_stats["mean"],
                "private_std": private_stats["std"],
                "shared_private_cross_covariance_norm": cross_covariance_norm,
                "shared_effective_rank": shared_stats["effective_rank"],
                **redundancy_stats,
            }

        _, completion_losses, completion_counts = self._completion_losses(output)
        diagnostics["completion"] = {
            modality: {
                "loss": float(completion_losses[modality].detach().item()),
                "target_count": int(completion_counts[modality].item()),
            }
            for modality in MODALITIES
        }
        for index, modality in enumerate(MODALITIES):
            completed_stats = stats(output["completed_shared_by_modality"][modality])
            natural_observed = output["natural_availability_mask"][:, index]
            diagnostics["completed_shared_views"][modality] = {
                "natural_observed_count": int(natural_observed.sum().item()),
                "naturally_completed_count": int((~natural_observed).sum().item()),
                "completion_target_count": int(completion_counts[modality].item()),
                "mean": completed_stats["mean"],
                "std": completed_stats["std"],
                "effective_rank": completed_stats["effective_rank"],
            }
        return diagnostics

    def compute_loss(self, data, output):
        rec = self._reconstruction_loss(data, output)
        align_correlation, align_variance, align_redundancy = self._alignment_components(output)
        align = (
            align_correlation
            + self.alignment_variance_weight * align_variance
            + self.alignment_redundancy_weight * align_redundancy
        )
        orth = self._orthogonality_loss(output)
        completion, completion_by_modality, completion_counts = self._completion_losses(output)
        total = (
            self.lambda_rec * rec
            + self.lambda_align * align
            + self.lambda_orth * orth
            + self.lambda_completion * completion
        )
        items = {
            "rec_loss": rec.detach(),
            "align_loss": align.detach(),
            "align_correlation_loss": align_correlation.detach(),
            "align_variance_loss": align_variance.detach(),
            "align_redundancy_loss": align_redundancy.detach(),
            "orth_loss": orth.detach(),
            "completion_loss": completion.detach(),
            "total_loss": total.detach(),
        }
        for modality in MODALITIES:
            items[f"completion_loss_{modality}"] = completion_by_modality[modality].detach()
            items[f"completion_target_count_{modality}"] = completion_counts[modality].detach()
        return total, items


class SharedPrivateCompletionTrainer:
    """Trainer that uses all naturally incomplete participants."""

    def __init__(self, args, modality_feature_indices):
        self.batch_size = int(args.batch_size)
        self.accum_iter = int(args.accum_iter)
        self.max_epochs = int(args.max_epochs)
        self.lr = args.lr if args.lr is not None else args.blr * self.batch_size / 64
        self.min_lr = float(args.min_lr)
        self.warmup_epochs = int(args.warmup_epochs)
        self.weight_decay = float(args.weight_decay)
        requested_device = str(args.device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        private_dims = {
            "host": int(args.private_dim_host),
            "endocrine": int(args.private_dim_endocrine),
            "brain": int(args.private_dim_brain),
        }
        self.model = SharedPrivateCompletionModel(
            modality_feature_indices=modality_feature_indices,
            shared_dim=int(args.shared_dim),
            private_dims=private_dims,
            hidden_dim=int(args.hidden_dim),
            lambda_rec=float(args.lambda_rec),
            lambda_align=float(args.lambda_align),
            lambda_orth=float(args.lambda_orth),
            lambda_completion=float(args.lambda_completion),
            alignment_variance_weight=float(args.alignment_variance_weight),
            alignment_redundancy_weight=float(args.alignment_redundancy_weight),
            variance_floor=float(args.variance_floor),
        ).to(self.device)
        self.dropout_probabilities = torch.tensor(
            [args.dropout_host, args.dropout_endocrine, args.dropout_brain],
            dtype=torch.float32,
            device=self.device,
        )
        self.norm_parameters = None
        self.last_dropout_summary = None

    def _prepare_tensor(self, data):
        data = data.detach().clone().float().cpu()
        observed = torch.isfinite(data)
        if self.norm_parameters is None:
            means = torch.zeros(data.shape[1], dtype=torch.float64)
            stds = torch.ones(data.shape[1], dtype=torch.float64)
            for index in range(data.shape[1]):
                values = data[observed[:, index], index].double()
                if values.numel() == 0:
                    continue
                means[index] = values.mean()
                std = values.std(unbiased=False)
                if torch.isfinite(std) and std > Z_SCORE_EPS:
                    stds[index] = std
            self.norm_parameters = {"mean": means.numpy(), "std": stds.numpy()}
        means = torch.as_tensor(self.norm_parameters["mean"], dtype=data.dtype)
        stds = torch.as_tensor(self.norm_parameters["std"], dtype=data.dtype)
        data = (data - means) / stds
        return torch.nan_to_num(data), observed

    def _sample_artificial_dropout(self, natural_availability):
        batch_size, num_modalities = natural_availability.shape
        dropout = torch.zeros_like(natural_availability, dtype=torch.bool)
        for row in range(batch_size):
            available = torch.where(natural_availability[row])[0]
            if available.numel() < 2:
                continue
            selected = torch.rand(num_modalities, device=self.device) < self.dropout_probabilities
            selected &= natural_availability[row]
            if selected.all():
                keep = available[torch.randint(available.numel(), (1,), device=self.device)]
                selected[keep] = False
            dropout[row] = selected
        return dropout

    def dropout_supervision_summary(self, observed_mask):
        """Expected supervised target counts under the configured dropout rates."""
        observed_mask = observed_mask.to(device=self.device, dtype=torch.bool)
        availability = torch.stack(
            [observed_mask[:, indices].any(dim=1) for indices in self.model.modality_feature_indices.values()],
            dim=1,
        )
        counts = availability.sum(dim=1)
        expected = {}
        for index, modality in enumerate(MODALITIES):
            eligible = availability[:, index] & (counts >= 2)
            # For a row with k available modalities, the all-dropped event is
            # corrected by randomly retaining one modality.
            all_selected_probability = torch.ones_like(counts, dtype=torch.float32)
            for other_index in range(len(MODALITIES)):
                all_selected_probability = all_selected_probability * torch.where(
                    availability[:, other_index],
                    self.dropout_probabilities[other_index],
                    torch.ones_like(all_selected_probability),
                )
            probabilities = torch.where(
                eligible,
                self.dropout_probabilities[index]
                - all_selected_probability / counts.clamp_min(1),
                torch.zeros_like(counts, dtype=torch.float32),
            )
            expected[modality] = {
                "available_participants": int(availability[:, index].sum().item()),
                "eligible_participants": int(eligible.sum().item()),
                "expected_completion_targets": float((eligible.float() * probabilities).sum().item()),
            }
        return expected

    def _train_epoch(self, data, observed_mask, optimizer, scaler, epoch):
        dataset = TensorDataset(data, observed_mask)
        loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size)
        self.model.train()
        loss_sums = {"total_loss": 0.0, "rec_loss": 0.0, "align_loss": 0.0,
                     "orth_loss": 0.0, "completion_loss": 0.0,
                     "align_redundancy_loss": 0.0}
        n_batches = 0
        completion_sums = {modality: 0.0 for modality in MODALITIES}
        completion_counts = {modality: 0 for modality in MODALITIES}
        optimizer.zero_grad()
        for step, (samples, masks) in enumerate(loader):
            if step % self.accum_iter == 0:
                adjust_learning_rate(
                    optimizer,
                    step / max(len(loader), 1) + epoch,
                    self.lr,
                    self.min_lr,
                    self.max_epochs,
                    self.warmup_epochs,
                )
            samples = samples.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).bool()
            availability = torch.stack(
                [masks[:, indices].any(dim=1) for indices in self.model.modality_feature_indices.values()],
                dim=1,
            )
            artificial_dropout = self._sample_artificial_dropout(availability)
            output = self.model(samples, masks, artificial_dropout)
            loss, loss_items = self.model.compute_loss(samples, output)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}")
            for key in loss_sums:
                loss_sums[key] += float(loss_items[key].item())
            n_batches += 1
            for modality in MODALITIES:
                count = int(loss_items[f"completion_target_count_{modality}"].item())
                completion_counts[modality] += count
                completion_sums[modality] += (
                    float(loss_items[f"completion_loss_{modality}"].item()) * count
                )
            scaler(
                loss / self.accum_iter,
                optimizer,
                parameters=self.model.parameters(),
                update_grad=(step + 1) % self.accum_iter == 0 or (step + 1) == len(loader),
            )
            if (step + 1) % self.accum_iter == 0 or (step + 1) == len(loader):
                optimizer.zero_grad()
        history = {key: value / max(n_batches, 1) for key, value in loss_sums.items()}
        history["completion_target_counts"] = completion_counts
        history["completion_losses"] = {
            modality: completion_sums[modality] / completion_counts[modality]
            if completion_counts[modality] else 0.0
            for modality in MODALITIES
        }
        return history

