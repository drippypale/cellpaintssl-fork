import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Sampler
from source.jump_data import custom_collate_fn
import source.augment as au
from source import SimCLR, get_jump_dataloaders
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import source.eval as evl


class BalancedBatchSampler(Sampler):
    """Ensures each batch contains samples from multiple domains"""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Fast initialization: get domain information without loading images
        self.domain_indices = self._get_domain_indices_fast()

        # Shuffle within each domain
        for domain in self.domain_indices:
            if self.shuffle:
                np.random.shuffle(self.domain_indices[domain])

        self.num_domains = len(self.domain_indices)
        self.domain_keys = list(self.domain_indices.keys())

        print(f"📊 [BALANCED_SAMPLER] Initialized with {self.num_domains} domains:")
        for domain, indices in self.domain_indices.items():
            print(f"  - Domain {domain}: {len(indices)} samples")

    def _get_domain_indices_fast(self):
        """Get domain indices without loading images - much faster"""
        domain_indices = {}

        # Check if this is a JUMP dataset with pre-computed domain indices
        if hasattr(self.dataset, "domain_indices"):
            # Use pre-computed domain indices - fastest method
            print("  🚀 Using pre-computed domain indices (fastest method)...")
            return self.dataset.domain_indices.copy()

        # Check if this is a JUMP dataset with domain labels
        elif hasattr(self.dataset, "batch_to_index"):
            # For JUMP datasets, we can get domain info from the underlying dataset
            print("  🚀 Using fast domain indexing for JUMP dataset...")

            # Get the original dataset from the subset
            if hasattr(self.dataset, "subset") and hasattr(
                self.dataset.subset, "dataset"
            ):
                original_dataset = self.dataset.subset.dataset
                batch_to_index = original_dataset.batch_to_index

                # Get the indices from the subset
                subset_indices = self.dataset.subset.indices

                # Group by domain using the original metadata
                for subset_idx, original_idx in enumerate(subset_indices):
                    try:
                        # Get metadata from original dataset without loading image
                        row = original_dataset.metadata_df.iloc[original_idx]
                        batch = row["batch"]
                        domain = batch_to_index.get(batch, 0)

                        if domain not in domain_indices:
                            domain_indices[domain] = []
                        domain_indices[domain].append(subset_idx)
                    except Exception as e:
                        print(f"⚠️  Error processing index {subset_idx}: {e}")
                        continue
            else:
                print(
                    "  ⚠️  Could not access original dataset, falling back to slow method..."
                )
                return self._get_domain_indices_slow()
        else:
            print("  ⚠️  Not a JUMP dataset, using slow method...")
            return self._get_domain_indices_slow()

        return domain_indices

    def _get_domain_indices_slow(self):
        """Fallback method that loads images to get domain labels"""
        print("  🐌 Using slow domain indexing (loading images)...")
        domain_indices = {}

        for idx in range(len(self.dataset)):
            try:
                # Handle different dataset formats
                item = self.dataset[idx]
                if len(item) == 3:
                    # JUMP format: (views, metadata, domain_label)
                    _, _, domain_label = item
                elif len(item) == 2:
                    # Standard format: (data, domain_label)
                    _, domain_label = item
                else:
                    print(
                        f"⚠️  Unexpected item format at index {idx}: {len(item)} elements"
                    )
                    continue

                domain = (
                    domain_label.item()
                    if torch.is_tensor(domain_label)
                    else domain_label
                )
                if domain not in domain_indices:
                    domain_indices[domain] = []
                domain_indices[domain].append(idx)
            except Exception as e:
                print(f"⚠️  Error processing index {idx}: {e}")
                continue

        return domain_indices

    def __iter__(self):
        k = len(self.domain_keys)
        per_dom = max(1, self.batch_size // k)  # quota per domain
        remainder = self.batch_size - per_dom * k

        # Initialize pointers for each domain
        domain_ptr = {d: 0 for d in self.domain_keys}

        while True:
            batch = []

            # Take per_dom samples from each domain
            for d in self.domain_keys:
                start = domain_ptr[d]
                end = start + per_dom
                if end > len(self.domain_indices[d]):
                    return  # stop when any domain exhausted
                batch.extend(self.domain_indices[d][start:end])
                domain_ptr[d] = end

            # Distribute remainder to first few domains
            for d in self.domain_keys[:remainder]:
                ptr = domain_ptr[d]
                if ptr >= len(self.domain_indices[d]):
                    return
                batch.append(self.domain_indices[d][ptr])
                domain_ptr[d] = ptr + 1

            yield batch

    def __len__(self):
        k = len(self.domain_keys)
        per_dom = max(1, self.batch_size // k)
        # Number of full batches limited by the smallest domain
        min_len = min(len(v) for v in self.domain_indices.values())
        batches_by_quota = min_len // per_dom
        return max(0, batches_by_quota)


class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return GradReverseFn.apply(x, self.lambd)


class SimCLRWithGRL(SimCLR):
    def __init__(
        self,
        num_domains: int,
        adv_lambda: float = 1.0,
        domain_loss_weight: float = 1.0,
        domain_hidden: int = 128,
        freeze_encoder: bool = True,
        adapter_hidden: int = 384,
        adapter_scale: float = 0.1,
        **kwargs,
    ):
        # Pop custom kwargs not recognized by base SimCLR
        projector_lr = kwargs.pop("projector_lr", 1e-4)
        head_lr_scale = kwargs.pop("head_lr_scale", 0.5)
        head_update_every = kwargs.pop("head_update_every", 1)
        lambda_delay_frac = kwargs.pop("lambda_delay_frac", 0.2)
        domain_head_dropout = kwargs.pop("domain_head_dropout", 0.1)

        super().__init__(**kwargs)
        self.num_domains = int(num_domains)
        self.adv_lambda = float(adv_lambda)
        self.domain_loss_weight = domain_loss_weight
        self.freeze_encoder = bool(freeze_encoder)

        # Attach custom hparams to both self and self.hparams for downstream access
        self.projector_lr = float(projector_lr)
        self.head_lr_scale = float(head_lr_scale)
        self.head_update_every = int(head_update_every)
        self.lambda_delay_frac = float(lambda_delay_frac)
        self.domain_head_dropout = float(domain_head_dropout)
        if hasattr(self, "hparams"):
            try:
                self.hparams.projector_lr = self.projector_lr
                self.hparams.head_lr_scale = self.head_lr_scale
                self.hparams.head_update_every = self.head_update_every
                self.hparams.lambda_delay_frac = self.lambda_delay_frac
                self.hparams.domain_head_dropout = self.domain_head_dropout
            except Exception:
                pass

        # Track best leakage trough
        self.best_batch_auc = float("inf")
        self.best_nsb_at_auc = 0.0

        # Initialize GRL with the user-specified lambda
        self.grl = GradientReversal(lambd=self.adv_lambda)

        # Residual adapter after backbone (trainable even if encoder is frozen)
        embed_dim = self.mlp[0].in_features  # backbone output dimension
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_hidden, embed_dim),
        )
        self.adapter_scale = float(adapter_scale)

        # Domain head now takes the adapter output (backbone+adapter) as input
        adapter_in_dim = embed_dim
        print(f"  🔧 Domain head input dimension: {adapter_in_dim}")
        print(f"  🔧 Adapter targets backbone dim (embed_dim): {embed_dim}")
        self.domain_head = nn.Sequential(
            nn.Linear(adapter_in_dim, domain_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.domain_head_dropout),
            nn.Linear(domain_hidden, self.num_domains),
        )

        if self.freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # Ensure adapter and domain head are trainable
        for p in self.adapter.parameters():
            p.requires_grad = True
        for p in self.domain_head.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        # Parameter groups with different learning rates
        adapter_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("adapter.") and p.requires_grad
        ]
        head_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("domain_head.") and p.requires_grad
        ]
        projector_params = [
            p
            for n, p in self.named_parameters()
            if n.startswith("mlp.") and p.requires_grad
        ]

        projector_lr = float(
            getattr(self.hparams, "projector_lr", getattr(self, "projector_lr", 1e-4))
        )
        head_lr_scale = float(
            getattr(self.hparams, "head_lr_scale", getattr(self, "head_lr_scale", 0.5))
        )
        head_lr = projector_lr * head_lr_scale

        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "adapter",
                    "params": adapter_params,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "name": "head",
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "name": "projector",
                    "params": projector_params,
                    "lr": projector_lr,
                    "weight_decay": self.hparams.weight_decay,
                },
            ]
        )
        lr_scheduler = self._get_lr_scheduler(optimizer)
        # Step LR every training step to ensure warmup works within first epoch
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "lr-AdamW",
            },
        }

    def _get_lr_scheduler(self, optimizer):
        from source import warmup_scheduler

        return warmup_scheduler.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            warmup_start_lr=1e-6,
            max_epochs=self.hparams.max_epochs,
            eta_min=self.hparams.lr_final_value,
        )

    def on_before_optimizer_step(self, optimizer):
        """Optionally update head LR every N steps by zeroing it on skipped steps."""
        try:
            update_every = int(
                getattr(
                    self.hparams,
                    "head_update_every",
                    getattr(self, "head_update_every", 1),
                )
            )
            projector_lr = float(
                getattr(
                    self.hparams, "projector_lr", getattr(self, "projector_lr", 1e-4)
                )
            )
            head_lr_scale = float(
                getattr(
                    self.hparams, "head_lr_scale", getattr(self, "head_lr_scale", 0.5)
                )
            )
            base_head_lr = projector_lr * head_lr_scale
            for group in optimizer.param_groups:
                name = group.get("name", "")
                if name == "head":
                    if update_every > 1 and (self.global_step % update_every) != 0:
                        group["lr"] = 0.0
                    else:
                        group["lr"] = base_head_lr
        except Exception:
            pass

    def _forward_embeddings(self, imgs):
        # imgs: concatenated tensor of shape [2*B, C, H, W]
        backbone_feats = self.backbone(imgs)
        # Apply residual adapter
        adapted = backbone_feats + self.adapter_scale * self.adapter(backbone_feats)
        # Penultimate projector features: Linear -> ReLU
        penultimate = self.mlp[1](self.mlp[0](adapted))
        # Final contrastive representation
        feats = self.mlp[2](penultimate)
        return feats, penultimate, adapted

    def _simclr_loss_and_metrics(self, feats, mode: str):
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Log SimCLR metrics (match base class behavior)
        self.log(mode + "_loss", nll, prog_bar=(mode == "train"))
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def _domain_loss_and_metrics(self, features_for_domain, domain_targets, mode: str):
        # Apply GRL before the domain head on the adapter output
        rev = self.grl(features_for_domain)
        logits = self.domain_head(rev)
        loss = F.cross_entropy(logits, domain_targets, label_smoothing=0.1)

        # Compute domain accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == domain_targets).float().mean()
            self.log(mode + "_domain_acc", acc, prog_bar=(mode == "train"))

        # Compute AUC if possible
        auc_value = float("nan")
        try:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                y_true = domain_targets.detach().cpu().numpy()

                # Check if we have multiple classes and enough samples
                unique_classes = np.unique(y_true)
                if len(unique_classes) > 1 and len(y_true) > 1 and self.num_domains > 1:
                    if self.num_domains == 2:
                        # Binary AUC expects scores for the positive class
                        auc_value = roc_auc_score(y_true, probs[:, 1])
                    else:
                        # Multi-class AUC
                        try:
                            auc_value = roc_auc_score(
                                y_true, probs, multi_class="ovr", average="macro"
                            )
                        except ValueError:
                            # Fallback to micro average
                            auc_value = roc_auc_score(
                                y_true, probs, multi_class="ovr", average="micro"
                            )
                else:
                    # Not enough diversity for AUC calculation
                    auc_value = float("nan")
        except Exception as e:
            # Log the specific error for debugging
            if (
                mode == "train"
                and hasattr(self, "current_epoch")
                and self.current_epoch % 5 == 0
            ):
                print(f"  ⚠️  AUC calculation failed: {e}")
            auc_value = float("nan")

        self.log(mode + "_domain_auc", auc_value, prog_bar=(mode == "train"))
        self.log(mode + "_domain_ce", loss)

        # Log additional domain metrics for debugging
        if (
            mode == "train"
            and hasattr(self, "current_epoch")
            and self.current_epoch % 5 == 0
        ):
            with torch.no_grad():
                # Log domain distribution
                domain_counts = torch.bincount(
                    domain_targets, minlength=self.num_domains
                )
                print(f"  📊 Domain distribution: {domain_counts.cpu().numpy()}")
                print(f"  📊 Domain accuracy: {acc:.3f}, AUC: {auc_value}")

                # Log detailed metrics to wandb
                if hasattr(self.logger, "experiment") and hasattr(
                    self.logger.experiment, "log"
                ):
                    # Log domain distribution as histogram
                    domain_dist = domain_counts.cpu().numpy()
                    for i, count in enumerate(domain_dist):
                        self.log(f"{mode}_domain_{i}_count", count, prog_bar=False)

                    # Log domain balance metric (entropy of distribution)
                    if count > 0:
                        probs = domain_dist / domain_dist.sum()
                        entropy = -np.sum(probs * np.log(probs + 1e-8))
                        self.log(f"{mode}_domain_entropy", entropy, prog_bar=False)

        return loss

    def _update_grl_lambda(self, batch_idx):
        """Update GRL lambda with delayed ramp: 0 for first frac, then Ganin to target."""
        if hasattr(self, "trainer") and self.trainer is not None:
            current_step = (
                self.current_epoch * self.trainer.num_training_batches + batch_idx
            )
            total_steps = self.hparams.max_epochs * self.trainer.num_training_batches
            p = current_step / max(1, total_steps)
            delay = float(
                getattr(
                    self.hparams,
                    "lambda_delay_frac",
                    getattr(self, "lambda_delay_frac", 0.2),
                )
            )
            if p < delay:
                lam = 0.0
            else:
                p2 = (p - delay) / max(1e-8, (1.0 - delay))
                lam = (2.0 / (1.0 + np.exp(-10 * p2)) - 1.0) * self.adv_lambda
            self.grl.lambd = lam
            if batch_idx % 10 == 0:
                print(
                    f"  [GRL] Epoch {self.current_epoch}, Batch {batch_idx}/{self.trainer.num_training_batches}, "
                    f"Progress: {p:.3f}, Lambda: {lam:.3f}"
                )

    def training_step(self, batch, batch_idx):
        # Update GRL lambda with schedule
        self._update_grl_lambda(batch_idx)

        # Progress tracking
        if batch_idx % 5 == 0:  # Print every 5 batches
            print(
                f"  [TRAIN] Epoch {self.current_epoch}, Batch {batch_idx}/{self.trainer.num_training_batches}"
            )

        imgs, _, domain_labels = batch
        # imgs is a list of two views; concatenate across batch dimension
        imgs = torch.cat(imgs, dim=0)

        feats, penultimate, adapted = self._forward_embeddings(imgs)

        # SimCLR loss and metrics
        simclr_loss = self._simclr_loss_and_metrics(feats, mode="train")

        # Domain targets repeated for both views
        domain_targets = (
            domain_labels.detach().clone().to(device=self.device, dtype=torch.long)
        )
        domain_targets = torch.cat([domain_targets, domain_targets], dim=0)

        # Domain loss on adapter output (export layer)
        domain_loss = self._domain_loss_and_metrics(
            adapted, domain_targets, mode="train"
        )

        # Combine losses without additional weighting (lambda is in GRL)
        total_loss = simclr_loss + self.domain_loss_weight * domain_loss
        self.log("train_total_loss", total_loss, prog_bar=True)

        # Log GRL lambda for tracking
        self.log("train_grl_lambda", self.grl.lambd, prog_bar=False)

        # Log detailed progress every 10 batches
        if batch_idx % 10 == 0:
            print(
                f"    [LOSSES] SimCLR: {simclr_loss:.4f}, Domain: {domain_loss:.4f}, Total: {total_loss:.4f}"
            )
            print(f"    [GRL] Current Lambda: {self.grl.lambd:.3f}")

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Progress tracking for validation
        if batch_idx % 2 == 0:  # Print every 2 validation batches
            print(f"  [VAL] Epoch {self.current_epoch}, Batch {batch_idx}")

        imgs, _, domain_labels = batch
        imgs = torch.cat(imgs, dim=0)
        feats, penultimate, adapted = self._forward_embeddings(imgs)
        _ = self._simclr_loss_and_metrics(feats, mode="val")
        domain_targets = (
            domain_labels.detach().clone().to(device=self.device, dtype=torch.long)
        )
        domain_targets = torch.cat([domain_targets, domain_targets], dim=0)
        _ = self._domain_loss_and_metrics(adapted, domain_targets, mode="val")

    def _mini_eval_and_log(self):
        """Run a lightweight validation-style eval on a few val batches and log metrics."""
        try:
            val_dls = getattr(self.trainer, "val_dataloaders", None)
            if val_dls is None:
                print("⚠️  No val_dataloaders available for mini-eval")
                return
            # Support both a single DataLoader or a list/tuple of loaders
            if isinstance(val_dls, (list, tuple)):
                if len(val_dls) == 0:
                    print("⚠️  Empty val_dataloaders for mini-eval")
                    return
                val_loader = val_dls[0]
            else:
                val_loader = val_dls

            # Label alignment note
            print(
                "🔎 Label alignment: domain_labels source column = 'batch'; batch leakage column = 'batch'"
            )

            self.eval()
            device = self.device
            rows = []
            max_batches = 10  # sample more batches to improve diversity
            batches_done = 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        imgs, metadata_collated, _domain = batch
                    except Exception:
                        try:
                            imgs, metadata_collated = batch
                            _domain = None
                        except Exception:
                            break

                    view = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
                    view = view.to(device)

                    backbone_feats = self.backbone(view)
                    adapted = backbone_feats + self.adapter_scale * self.adapter(
                        backbone_feats
                    )
                    embs = adapted.detach().cpu().numpy()

                    B = embs.shape[0]
                    # metadata_collated is a dict of lists per custom_collate_fn
                    for i in range(B):
                        meta = {}
                        if isinstance(metadata_collated, dict):
                            for k in ("batch", "plate", "well", "compound", "target"):
                                try:
                                    seq = metadata_collated.get(k, None)
                                    meta[k] = (
                                        seq[i]
                                        if isinstance(seq, list) and len(seq) > i
                                        else None
                                    )
                                except Exception:
                                    meta[k] = None
                        elif (
                            isinstance(metadata_collated, (list, tuple))
                            and len(metadata_collated) > i
                        ):
                            md = metadata_collated[i]
                            if isinstance(md, dict):
                                meta = md

                        row = {
                            "batch": meta.get("batch", None),
                            "plate": meta.get("plate", None),
                            "well": meta.get("well", None),
                            # map compound to perturbation_id for eval compatibility
                            "perturbation_id": meta.get("compound", None),
                            "target": meta.get("target", None),
                        }
                        for j, v in enumerate(embs[i]):
                            row[f"emb{j}"] = float(v)
                        rows.append(row)

                    batches_done += 1

                    # Early stop if we have enough diversity
                    if batches_done >= max_batches:
                        break
                    if len(rows) >= 200:
                        df_tmp = pd.DataFrame(rows)
                        if (
                            df_tmp.get("batch").nunique(dropna=True) >= 2
                            and df_tmp.get("perturbation_id").nunique(dropna=True) >= 2
                        ):
                            break

            if len(rows) < 20:
                return

            df = pd.DataFrame(rows)
            feature_cols = [c for c in df.columns if c.startswith("emb")]

            # For alignment visibility, print a small sample
            try:
                u_batches = df.get("batch").dropna().astype(str).unique().tolist()[:5]
                print(f"  ↳ Sample batches (minieval): {u_batches}")
            except Exception:
                pass

            # Batch leakage proxies (only if >=2 batch classes)
            batch_auc_val = None
            try:
                if df.get("batch").nunique(dropna=True) >= 2:
                    batch_metrics = evl.batch_classification(df)
                    self.log(
                        "val_batch_MCC", float(batch_metrics.get("MCC", float("nan")))
                    )
                    batch_auc_val = float(batch_metrics.get("ROC_AUC", float("nan")))
                    self.log(
                        "val_batch_ROC_AUC",
                        batch_auc_val,
                    )
                else:
                    print(
                        "⚠️  mini-eval: only one batch class observed; skipping batch ROC_AUC/MCC"
                    )
            except Exception as e:
                print(f"⚠️  batch classification eval failed: {e}")

            # NSB perturbation accuracy
            nsb_acc_val = None
            if (
                "perturbation_id" in df.columns
                and df["perturbation_id"].notnull().any()
            ):
                try:
                    X = StandardScaler().fit_transform(df[feature_cols].to_numpy())
                    y = LabelEncoder().fit_transform(
                        df["perturbation_id"].astype(str).values
                    )
                    batches = (
                        df["batch"].astype(str).values
                        if "batch" in df.columns
                        else np.array([""] * len(df))
                    )
                    if len(np.unique(batches)) < 2:
                        print(
                            "⚠️  mini-eval: only one batch label; NSB may be ill-posed"
                        )
                    ypred_nsb = evl.nearest_neighbor_classifier_NSBW(
                        X, y, mode="NSB", batches=batches, metric="cosine"
                    )
                    nsb_acc = (ypred_nsb == y).mean()
                    nsb_acc_val = float(nsb_acc)
                    self.log("val_pert_acc_NSB", nsb_acc_val)
                except Exception as e:
                    print(f"⚠️  NSB perturbation eval failed: {e}")

            # Perturbation mAP (aggregated)
            try:
                # Keep only valid perturbation IDs with at least 2 samples
                df_valid = df[
                    df["perturbation_id"].notnull()
                    & (df["perturbation_id"].astype(str) != "")
                ].copy()
                vc = df_valid["perturbation_id"].value_counts()
                keep_ids = set(vc[vc >= 2].index.tolist())
                df_valid = df_valid[df_valid["perturbation_id"].isin(keep_ids)].copy()
                if df_valid["perturbation_id"].nunique() < 2:
                    print(
                        "⚠️  mAP eval skipped: not enough perturbations with >=2 samples"
                    )
                else:
                    agg = (
                        df_valid.groupby(["perturbation_id"])
                        .mean(numeric_only=True)
                        .reset_index()
                    )
                    # Standardize features for stability
                    Xagg = StandardScaler().fit_transform(agg[feature_cols].to_numpy())
                    agg_std = agg.copy()
                    for j, c in enumerate(feature_cols):
                        agg_std[c] = Xagg[:, j]
                    pr_k, pr_dist = evl.calculate_precision_recall(
                        agg_std, feature_cols, label_col="perturbation_id"
                    )
                    if "mAP" in pr_dist.columns:
                        mAP_val = float(pr_dist["mAP"].iloc[0])
                        if not np.isnan(mAP_val):
                            self.log("val_pert_mAP", mAP_val)
                        else:
                            print("⚠️  mAP returned NaN; skipping log")
                    else:
                        print("⚠️  mAP column not found in precision-recall output")
            except Exception as e:
                print(f"⚠️  Perturbation mAP eval failed: {e}")

            # Early-stop checkpointing at leakage trough
            try:
                if (
                    batch_auc_val is not None
                    and nsb_acc_val is not None
                    and not np.isnan(batch_auc_val)
                ):
                    improved_auc = batch_auc_val < (self.best_batch_auc - 1e-6)
                    nsb_not_worse = nsb_acc_val >= (self.best_nsb_at_auc - 1e-6)
                    if improved_auc and nsb_not_worse:
                        self.best_batch_auc = batch_auc_val
                        self.best_nsb_at_auc = nsb_acc_val
                        ckpt_dir = self.trainer.default_root_dir or "."
                        fname = f"earlystop_epoch{self.current_epoch:02d}_auc{batch_auc_val:.4f}_nsb{nsb_acc_val:.4f}.ckpt"
                        path = os.path.join(ckpt_dir, fname)
                        self.trainer.save_checkpoint(path)
                        print(f"💾 Saved early-stop checkpoint at AUC trough: {path}")
                        # Log best-so-far values
                        self.log("val_best_batch_ROC_AUC", self.best_batch_auc)
                        self.log("val_best_nsb_at_auc", self.best_nsb_at_auc)
            except Exception as e:
                print(f"⚠️  Early-stop checkpointing failed: {e}")
        except Exception as e:
            print(f"⚠️  Mini-eval logging failed: {e}")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        print(f"✅ Epoch {self.current_epoch} training completed")
        print(f"  - Final GRL lambda: {self.adv_lambda:.3f}")

        # Log epoch-level metrics to wandb
        if hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "log"
        ):
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log("train_learning_rate", current_lr, prog_bar=False)
            print(f"  - Current LR: {current_lr:.6e}")
            self.log("train_epoch", self.current_epoch, prog_bar=False)

        # Always run mini-eval here to guarantee per-epoch logging
        self._mini_eval_and_log()
        print()

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        print(f"✅ Epoch {self.current_epoch} validation completed")

        if hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "log"
        ):
            self.log("val_epoch", self.current_epoch, prog_bar=False)

        # Also run the same mini-eval here (if/when val loop executes)
        self._mini_eval_and_log()
        print()


def main():
    parser = argparse.ArgumentParser()
    # Model and training arguments
    parser.add_argument("--arch", default="vit_small_patch16_224", type=str)
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0, 1], help="Number of GPUs to use"
    )
    parser.add_argument(
        "--submission_csv",
        type=str,
        required=True,
        help="Path to submission CSV with job paths",
    )
    parser.add_argument(
        "--images_base_path",
        type=str,
        default="/content/drive/MyDrive/jump_data/images",
        help="Base path where JUMP images are stored",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=1, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Checkpoint directory for logs and new checkpoints",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="Path to pretrained SimCLR checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--every_n_epochs", type=int, default=1, help="Save checkpoint every n epochs"
    )
    # LR scheduler parameters
    parser.add_argument("--warmup_epochs", type=int, default=1, help="LR warmup epochs")
    parser.add_argument(
        "--lr_final_value", type=float, default=1e-6, help="Cosine final LR"
    )
    # Adam parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    # Model parameters
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of contrastive projector output (128 to match pretrained checkpoint)",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--scale", type=float, default=None, help="Scale for RandomResizedCrop"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    # Adversarial domain head params
    parser.add_argument(
        "--adv_lambda",
        type=float,
        default=1.0,
        help="Weight for adversarial domain CE loss",
    )
    parser.add_argument(
        "--lambda_delay_frac",
        type=float,
        default=0.2,
        help="Fraction of total steps to keep GRL lambda at 0 before ramping",
    )
    parser.add_argument(
        "--domain_hidden", type=int, default=128, help="Hidden units in domain head"
    )
    parser.add_argument(
        "--domain_head_dropout",
        type=float,
        default=0.1,
        help="Dropout probability before domain head logits",
    )
    parser.add_argument(
        "--no_freeze_encoder",
        action="store_true",
        help="Do not freeze the encoder backbone",
    )
    # Adapter params
    parser.add_argument(
        "--adapter_hidden",
        type=int,
        default=384,
        help="Hidden units in residual adapter after backbone",
    )
    parser.add_argument(
        "--adapter_scale",
        type=float,
        default=0.1,
        help="Residual scale for adapter output (y = x + scale*f(x))",
    )
    parser.add_argument(
        "--projector_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the projector (mlp)",
    )
    parser.add_argument(
        "--head_lr_scale",
        type=float,
        default=0.5,
        help="Domain head LR as a scale of projector_lr (e.g., 0.5 means head is 0.5x)",
    )
    parser.add_argument(
        "--head_update_every",
        type=int,
        default=2,
        help="Update the domain head every N steps (set 1 to disable)",
    )
    # Data parameters
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to load (for debugging)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--balanced_batches",
        action="store_true",
        help="Use balanced batch sampling to ensure domain diversity",
    )
    # Wandb logging arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cellpaint-ssl-grl",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity/username (optional)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Custom run name for wandb (optional)",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags for wandb run",
    )
    parser.add_argument(
        "--domain_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the CE loss of the domain loss in the total loss",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 STARTING GRL JUMP TRAINING")
    print("=" * 80)
    print(f"Arguments: {vars(args)}")
    print()

    # Unpack
    submission_csv = args.submission_csv
    images_base_path = args.images_base_path
    gpus = args.gpus
    batch_size = int(args.batch_size / max(1, len(gpus)))
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    ckpt_path = args.ckpt_path
    pretrained_ckpt = args.pretrained_ckpt
    every_n_epochs = args.every_n_epochs
    lr = args.lr
    weight_decay = args.wd
    hidden_dim = args.hidden_dim
    temperature = args.temperature
    size = args.size
    scale = args.scale
    adv_lambda = args.adv_lambda
    domain_hidden = args.domain_hidden
    freeze_encoder = not args.no_freeze_encoder
    max_samples = args.max_samples
    train_ratio = args.train_ratio
    balanced_batches = args.balanced_batches
    warmup_epochs = args.warmup_epochs
    lr_final_value = args.lr_final_value
    adapter_hidden = args.adapter_hidden
    adapter_scale = args.adapter_scale
    domain_loss_weight = args.domain_loss_weight

    print("📋 CONFIGURATION:")
    print(f"  - Submission CSV: {submission_csv}")
    print(f"  - Images base path: {images_base_path}")
    print(f"  - Batch size: {batch_size} (effective: {batch_size * max(1, len(gpus))})")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Adversarial lambda: {adv_lambda}")
    print(f"  - Freeze encoder: {freeze_encoder}")
    print(f"  - Balanced batches: {balanced_batches}")
    print(f"  - Max samples: {max_samples}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print()

    # Seed and determinism
    print("🔧 SETTING UP ENVIRONMENT...")
    pl.seed_everything(args.seed)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"  - Checkpoint base directory: {ckpt_path}")
    print(f"  - Directory exists: {os.path.exists(ckpt_path)}")
    print(f"  - Directory writable: {os.access(ckpt_path, os.W_OK)}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"  - Device: {device}")
    print(f"  - Seed: {args.seed}")
    print()

    # Augmentations (as in SimCLR)
    print("🔄 SETTING UP AUGMENTATIONS...")
    # Note: These are the original Cell Painting means/stds, may need adjustment for JUMP
    means = [0.13849893, 0.18710597, 0.1586524, 0.15757588, 0.08674719]
    stds = [0.13005716, 0.15461144, 0.15929441, 0.16021383, 0.16686504]
    color_prob = 1
    base_transform = au.get_default_augs(color_prob, means, stds)
    transform = au.Augment(
        crop_transform=au.RandomCropWithCells(size=size, scale=scale),
        base_transforms=base_transform,
        n_views=2,
    )
    print(f"  - Image size: {size}")
    print(f"  - Scale: {scale}")
    print(f"  - Color prob: {color_prob}")
    print()

    # Create dataloaders with domain labels
    print("📊 LOADING JUMP DATASET...")
    print(f"  - Loading from: {submission_csv}")
    print(f"  - Images from: {images_base_path}")
    print(f"  - Max samples: {max_samples}")
    print(f"  - Train ratio: {train_ratio}")

    train_loader, val_loader, batch_to_index = get_jump_dataloaders(
        submission_csv=submission_csv,
        images_base_path=images_base_path,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        max_samples=max_samples,
        with_domain_labels=True,
    )

    # Optionally use balanced batch sampling for training
    if balanced_batches:
        print("⚖️  SETTING UP BALANCED BATCH SAMPLING...")
        # Get the training dataset from the dataloader
        train_dataset = train_loader.dataset
        balanced_sampler = BalancedBatchSampler(train_dataset, batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,  # Use batch_sampler instead of sampler
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,  # Use custom collate function for metadata
            # Note: batch_sampler handles drop_last internally by only yielding complete batches
        )
        print("  - Balanced batch sampling enabled")

    # Dataloaders ready... prints are above
    num_domains = len(batch_to_index)
    print(f"  - Number of domains: {num_domains}")
    print(f"  - Domain mapping: {batch_to_index}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print()

    # Explicit label alignment note
    print("🔎 Domain labels source column: 'batch'")
    print("🔎 Batch leakage evaluation column: 'batch'")

    # Setup wandb logging
    loggers = []
    if args.use_wandb:
        print("📊 SETTING UP WANDB LOGGING...")

        # Create run name if not provided
        if args.wandb_run_name is None:
            run_name = f"grl_jump_adv{adv_lambda}_bs{batch_size}_lr{lr}"
            if balanced_batches:
                run_name += "_balanced"
            args.wandb_run_name = run_name

        # Setup wandb logger
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            log_model=True,  # Log model checkpoints
        )

        # Log hyperparameters
        wandb_logger.log_hyperparams(
            {
                "arch": args.arch,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "lr": lr,
                "warmup_epochs": warmup_epochs,
                "lr_final_value": lr_final_value,
                "weight_decay": weight_decay,
                "hidden_dim": hidden_dim,
                "temperature": temperature,
                "adv_lambda": adv_lambda,
                "lambda_delay_frac": args.lambda_delay_frac,
                "domain_loss_wight": domain_loss_weight,
                "domain_hidden": domain_hidden,
                "domain_head_dropout": args.domain_head_dropout,
                "freeze_encoder": freeze_encoder,
                "balanced_batches": balanced_batches,
                "num_workers": num_workers,
                "train_ratio": train_ratio,
                "num_domains": num_domains,
                "adapter_hidden": adapter_hidden,
                "adapter_scale": adapter_scale,
                "projector_lr": args.projector_lr,
                "head_lr_scale": args.head_lr_scale,
                "head_update_every": args.head_update_every,
            }
        )

        loggers.append(wandb_logger)
        print(f"  - Project: {args.wandb_project}")
        print(f"  - Run name: {args.wandb_run_name}")
        print(f"  - Tags: {args.wandb_tags}")
        print()

    # Trainer
    print("🏃 SETTING UP TRAINER...")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(ckpt_path, "GRL_Jump_SimCLR"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        strategy="auto",
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,  # Log every 10 steps instead of default 50
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(ckpt_path, "GRL_Jump_SimCLR"),
                save_weights_only=True,
                save_top_k=3,  # Save top 3 checkpoints
                monitor="train_total_loss",
                mode="min",
                every_n_epochs=every_n_epochs,  # Save every n epochs from args
                save_last=True,  # Always save the last checkpoint
                filename="epoch_{epoch:02d}-{train_total_loss:.4f}",
            ),
            LearningRateMonitor("epoch"),
        ],
        gradient_clip_val=3.0,
        logger=loggers,  # Add wandb logger
    )
    trainer.logger._default_hp_metric = None
    print(f"  - Checkpoint dir: {os.path.join(ckpt_path, 'GRL_Jump_SimCLR')}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Gradient clip: 3.0")
    print()

    # Model
    print("🧠 CREATING MODEL...")
    print(f"  - Creating SimCLRWithGRL with hidden_dim={hidden_dim}")
    model = SimCLRWithGRL(
        num_domains=num_domains,
        adv_lambda=adv_lambda,
        domain_hidden=domain_hidden,
        freeze_encoder=freeze_encoder,
        adapter_hidden=adapter_hidden,
        adapter_scale=adapter_scale,
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        lr_final_value=lr_final_value,
        lr=lr,
        hidden_dim=hidden_dim,
        temperature=temperature,
        weight_decay=weight_decay,
        vit=args.arch,
        # pass new hparams through
        projector_lr=args.projector_lr,
        head_lr_scale=args.head_lr_scale,
        head_update_every=args.head_update_every,
        lambda_delay_frac=args.lambda_delay_frac,
        domain_head_dropout=args.domain_head_dropout,
    )
    print(f"  - Architecture: {args.arch}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Temperature: {temperature}")
    print(f"  - Domain hidden: {domain_hidden}")
    print(f"  - Freeze encoder: {freeze_encoder}")
    print(f"  - Adapter: hidden={adapter_hidden}, scale={adapter_scale}")

    # Debug MLP structure
    print(f"  - MLP structure:")
    print(f"    - mlp[0]: {model.mlp[0]} (output dim: {model.mlp[0].out_features})")
    print(f"    - mlp[1]: {model.mlp[1]}")
    print(f"    - mlp[2]: {model.mlp[2]} (output dim: {model.mlp[2].out_features})")
    print()

    # Sanity check: Test GRL gradient reversal
    print("🔍 TESTING GRL GRADIENT REVERSAL...")
    z = torch.randn(8, 16, requires_grad=True)
    grl = GradientReversal(lambd=0.5)
    loss = grl(z).sum()
    loss.backward()
    expected_grad = -0.5
    actual_grad = z.grad.mean().item()
    print(f"  - Expected gradient: {expected_grad}, Actual gradient: {actual_grad}")
    if abs(actual_grad - expected_grad) < 1e-5:
        print("  ✅ GRL gradient reversal working correctly")
    else:
        print("  ❌ GRL gradient reversal test failed")
        print(f"  - Difference: {abs(actual_grad - expected_grad)}")
    print()

    # Load pretrained SimCLR weights (backbone + projector). Allow missing for new head.
    print("📥 LOADING PRETRAINED CHECKPOINT...")
    print(f"  - Loading from: {pretrained_ckpt}")
    ckpt = torch.load(pretrained_ckpt, map_location="cpu")

    # Debug checkpoint structure
    print(
        f"  - Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'Not a dict'}"
    )

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("  - Using 'state_dict' key")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            print("  - Using 'model' key")
        else:
            state_dict = ckpt
            print("  - Using direct checkpoint dict")
    else:
        state_dict = ckpt
        print("  - Using checkpoint as state dict")

    print(
        f"  - State dict keys: {list(state_dict.keys())[:10]}..."
    )  # Show first 10 keys

    # Debug: Show MLP-related keys in checkpoint
    mlp_keys = [k for k in state_dict.keys() if "mlp" in k]
    print(f"  - MLP keys in checkpoint: {mlp_keys}")

    # Debug: Show expected vs actual MLP dimensions
    print(f"  - Expected MLP dimensions (hidden_dim={hidden_dim}):")
    print(
        f"    - mlp.0.weight: [4*{hidden_dim}, embed_dim] = [{4 * hidden_dim}, embed_dim]"
    )
    print(
        f"    - mlp.2.weight: [{hidden_dim}, 4*{hidden_dim}] = [{hidden_dim}, {4 * hidden_dim}]"
    )

    # Show actual dimensions from checkpoint
    if "mlp.0.weight" in state_dict:
        actual_dim0 = state_dict["mlp.0.weight"].shape
        print(f"  - Actual mlp.0.weight: {actual_dim0}")
    if "mlp.2.weight" in state_dict:
        actual_dim2 = state_dict["mlp.2.weight"].shape
        print(f"  - Actual mlp.2.weight: {actual_dim2}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Optional: print/load info for debugging
    print(f"  - Loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  - Missing keys: {missing[:5]}...")  # Show first 5 missing keys
    if unexpected:
        print(
            f"  - Unexpected keys: {unexpected[:5]}..."
        )  # Show first 5 unexpected keys
    print()

    # Fit
    print("🎯 STARTING TRAINING...")
    print("=" * 80)
    print(
        "Training will show progress every 5 batches with detailed logs every 10 batches"
    )
    print(
        "Look for: [TRAIN], [VAL], [GRL], [LOSSES] prefixes for different progress types"
    )
    print("=" * 80)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)

    # Check if checkpoints were saved
    checkpoint_dir = os.path.join(ckpt_path, "GRL_Jump_SimCLR")
    print(f"🔍 Checking checkpoint directory: {checkpoint_dir}")
    print(f"  - Directory exists: {os.path.exists(checkpoint_dir)}")

    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        print(f"📁 Checkpoints saved: {len(checkpoints)}")
        for ckpt in checkpoints:
            ckpt_path_full = os.path.join(checkpoint_dir, ckpt)
            ckpt_size = (
                os.path.getsize(ckpt_path_full) if os.path.exists(ckpt_path_full) else 0
            )
            print(f"  - {ckpt} ({ckpt_size} bytes)")

        # Also check for any other files in the directory
        all_files = os.listdir(checkpoint_dir)
        other_files = [f for f in all_files if not f.endswith(".ckpt")]
        if other_files:
            print(f"📄 Other files in checkpoint directory: {other_files}")
    else:
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print(
            "  - This might indicate a permission issue or the directory wasn't created"
        )

        # Check if the base directory exists
        print(f"  - Base directory exists: {os.path.exists(ckpt_path)}")
        if os.path.exists(ckpt_path):
            print(f"  - Base directory contents: {os.listdir(ckpt_path)}")


if __name__ == "__main__":
    main()
