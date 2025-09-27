import os
import argparse
import numpy as np
import math
import copy
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
        # Re-shuffle each epoch for fresh sampling
        if self.shuffle:
            for d in self.domain_indices:
                np.random.shuffle(self.domain_indices[d])
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
        # Progressive unfreezing + LLRD
        unfreeze_start_epoch: int = 5,
        unfreeze_every: int = 5,
        max_trainable_blocks: int = 2,
        llrd_decay: float = 0.65,
        backbone_lr_scale: float = 0.2,
        freeze_patch_embed: bool = True,
        # Triplet refinements
        triplet_use_memory: bool = False,
        triplet_memory_size: int = 4096,
        triplet_semi_hard: bool = True,
        **kwargs,
    ):
        # Pop custom kwargs not recognized by base SimCLR
        projector_lr = kwargs.pop("projector_lr", 1e-4)
        head_lr_scale = kwargs.pop("head_lr_scale", 0.5)
        head_update_every = kwargs.pop("head_update_every", 1)
        lambda_delay_frac = kwargs.pop("lambda_delay_frac", 0.2)
        domain_head_dropout = kwargs.pop("domain_head_dropout", 0.1)
        domain_label_key = kwargs.pop("domain_label_key", "batch")
        # Anti-forgetting weights
        distill_weight = kwargs.pop("distill_weight", 0.5)
        rel_kd_weight = kwargs.pop("rel_kd_weight", 0.1)
        l2sp_weight = kwargs.pop("l2sp_weight", 1e-3)
        adapter_anchor_weight = kwargs.pop("adapter_anchor_weight", 1e-4)
        # Triplet loss params
        triplet_weight = kwargs.pop("triplet_weight", 0.0)
        triplet_margin = kwargs.pop("triplet_margin", 0.2)
        triplet_metric = kwargs.pop("triplet_metric", "cosine")
        # XGBoost masking params for validation reporting
        xgb_mask_fraction = kwargs.pop("xgb_mask_fraction", 0.0)
        xgb_top_k_plot = kwargs.pop("xgb_top_k_plot", 40)
        xgb_random_state = kwargs.pop("xgb_random_state", 42)

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
        self.domain_label_key = str(domain_label_key)
        # Store anti-forgetting weights
        self.distill_weight = float(distill_weight)
        self.rel_kd_weight = float(rel_kd_weight)
        self.l2sp_weight = float(l2sp_weight)
        self.adapter_anchor_weight = float(adapter_anchor_weight)
        # Triplet
        self.triplet_weight = float(triplet_weight)
        self.triplet_margin = float(triplet_margin)
        self.triplet_metric = str(triplet_metric)
        # XGB masking params
        self.xgb_mask_fraction = float(xgb_mask_fraction)
        self.xgb_top_k_plot = int(xgb_top_k_plot)
        self.xgb_random_state = int(xgb_random_state)
        # Unfreezing & LLRD
        self.unfreeze_start_epoch = int(unfreeze_start_epoch)
        self.unfreeze_every = int(unfreeze_every)
        self.max_trainable_blocks = int(max_trainable_blocks)
        self.llrd_decay = float(llrd_decay)
        self.backbone_lr_scale = float(backbone_lr_scale)
        self.freeze_patch_embed = bool(freeze_patch_embed)
        # Triplet refinements
        self.triplet_use_memory = bool(triplet_use_memory)
        self.triplet_memory_size = int(triplet_memory_size)
        self.triplet_semi_hard = bool(triplet_semi_hard)
        if hasattr(self, "hparams"):
            try:
                self.hparams.projector_lr = self.projector_lr
                self.hparams.head_lr_scale = self.head_lr_scale
                self.hparams.head_update_every = self.head_update_every
                self.hparams.lambda_delay_frac = self.lambda_delay_frac
                self.hparams.domain_head_dropout = self.domain_head_dropout
                self.hparams.domain_label_key = self.domain_label_key
                self.hparams.distill_weight = self.distill_weight
                self.hparams.rel_kd_weight = self.rel_kd_weight
                self.hparams.l2sp_weight = self.l2sp_weight
                self.hparams.adapter_anchor_weight = self.adapter_anchor_weight
                self.hparams.triplet_weight = self.triplet_weight
                self.hparams.triplet_margin = self.triplet_margin
                self.hparams.triplet_metric = self.triplet_metric
                # Unfreezing & LLRD
                self.hparams.unfreeze_start_epoch = self.unfreeze_start_epoch
                self.hparams.unfreeze_every = self.unfreeze_every
                self.hparams.max_trainable_blocks = self.max_trainable_blocks
                self.hparams.llrd_decay = self.llrd_decay
                self.hparams.backbone_lr_scale = self.backbone_lr_scale
                self.hparams.freeze_patch_embed = self.freeze_patch_embed
                # Triplet memory
                self.hparams.triplet_use_memory = self.triplet_use_memory
                self.hparams.triplet_memory_size = self.triplet_memory_size
                self.hparams.triplet_semi_hard = self.triplet_semi_hard
                # XGB masking params
                self.hparams.xgb_mask_fraction = self.xgb_mask_fraction
                self.hparams.xgb_top_k_plot = self.xgb_top_k_plot
                self.hparams.xgb_random_state = self.xgb_random_state
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

        # Cache ViT structure and set initial trainable blocks
        self._prepare_backbone_blocks()
        self.current_trainable_blocks = (
            0
            if self.freeze_encoder
            else min(self.max_trainable_blocks, self._num_vit_blocks)
        )
        if self.current_trainable_blocks > 0:
            self._set_trainable_blocks(self.current_trainable_blocks)

        # Triplet memory buffers (lazy init)
        self._trip_mem_feats = None
        self._trip_mem_labels = None
        self._trip_mem_ptr = 0

        # Extended triplet memory for NSBP constraints
        self._trip_mem_batches = None
        self._trip_mem_perts = None
        # Global id maps to keep memory label ids stable across batches
        self._trip_target_to_id = {}
        self._trip_batch_to_id = {}
        self._trip_pert_to_id = {}

        # Teacher/backbone anchor placeholders (to be initialized after loading ckpt)
        self.teacher_backbone = None
        self.backbone_init_state = None

    def _batch_hard_triplet_loss(
        self, features_first_view, metadata_collated, mode: str
    ):
        """Compute batch-hard triplet loss over first-view features using 'target' labels.

        Args:
            features_first_view: Tensor [B, D] (no concat of views)
            metadata_collated: dict of lists with key 'target'
        Returns:
            loss: scalar tensor
        """
        weight = float(getattr(self, "triplet_weight", 0.0))
        if weight <= 0.0:
            return torch.tensor(0.0, device=features_first_view.device)

        targets_list = []
        try:
            targets_list = (
                metadata_collated.get("target", [])
                if isinstance(metadata_collated, dict)
                else []
            )
        except Exception:
            targets_list = []

        B = features_first_view.shape[0]
        if not isinstance(targets_list, list) or len(targets_list) != B:
            # Can't compute
            loss = torch.tensor(0.0, device=features_first_view.device)
            self.log(mode + "_triplet_loss", loss, prog_bar=False)
            return loss

        # Normalize target strings and gather batch/pert identifiers
        labels = ["" if (t is None) else str(t) for t in targets_list]
        try:
            batches_list = [str(x) for x in metadata_collated.get("batch", [""] * B)]
        except Exception:
            batches_list = [""] * B
        try:
            perts_list = [str(x) for x in metadata_collated.get("compound", [""] * B)]
        except Exception:
            perts_list = [""] * B

        # Build mask of valid entries (non-empty targets)
        valid_mask = torch.tensor(
            [
                label_str.strip() != "" and label_str.lower() != "nan"
                for label_str in labels
            ],
            device=features_first_view.device,
        )
        if valid_mask.sum().item() < 2:
            loss = torch.tensor(0.0, device=features_first_view.device)
            self.log(mode + "_triplet_loss", loss, prog_bar=False)
            return loss

        # Map string labels to per-batch ids for grouping within this batch
        uniq = {}
        mapped = []
        for label_str in labels:
            if label_str.strip() == "" or label_str.lower() == "nan":
                mapped.append(-1)
            else:
                if label_str not in uniq:
                    uniq[label_str] = len(uniq)
                mapped.append(uniq[label_str])
        y = torch.tensor(mapped, device=features_first_view.device, dtype=torch.long)

        # Stable global ids for memory lookups
        def _get_or_add_id(mapping: dict, key: str) -> int:
            if key not in mapping:
                mapping[key] = len(mapping)
            return mapping[key]

        y_global = torch.tensor(
            [
                _get_or_add_id(self._trip_target_to_id, s)
                if (s.strip() != "" and s.lower() != "nan")
                else -1
                for s in labels
            ],
            device=features_first_view.device,
            dtype=torch.long,
        )
        b_global = torch.tensor(
            [
                _get_or_add_id(self._trip_batch_to_id, s) if s.strip() != "" else -1
                for s in batches_list
            ],
            device=features_first_view.device,
            dtype=torch.long,
        )
        p_global = torch.tensor(
            [
                _get_or_add_id(self._trip_pert_to_id, s) if s.strip() != "" else -1
                for s in perts_list
            ],
            device=features_first_view.device,
            dtype=torch.long,
        )

        # Restrict to valid indices where label >=0
        idx = torch.nonzero(
            torch.logical_and(valid_mask, y >= 0), as_tuple=False
        ).squeeze(1)
        if idx.numel() < 2:
            loss = torch.tensor(0.0, device=features_first_view.device)
            self.log(mode + "_triplet_loss", loss, prog_bar=False)
            return loss

        X = features_first_view[idx]
        y_idx = y[idx]
        y_g = y_global[idx]
        b_g = b_global[idx]
        p_g = p_global[idx]

        # Need at least one class with count>=2 and at least 2 classes
        classes, counts = torch.unique(y_idx, return_counts=True)
        if (counts >= 2).sum().item() == 0 or classes.numel() < 2:
            loss = torch.tensor(0.0, device=features_first_view.device)
            self.log(mode + "_triplet_loss", loss, prog_bar=False)
            return loss

        # Compute normalized distances (cosine or L2 over normalized vectors)
        metric = getattr(self, "triplet_metric", "cosine")
        Xn = F.normalize(X, dim=1)
        if metric == "cosine":
            sim = torch.matmul(Xn, Xn.t())
            dist_in = (1.0 - sim).clamp_min(0.0)
        else:
            dist_in = torch.cdist(Xn, Xn, p=2)

        M = dist_in.shape[0]
        # NSBP masks: same target, different batch and different perturbation
        y_col = y_idx.view(-1, 1)
        same_t = y_col.eq(y_col.t())
        b_col = b_g.view(-1, 1)
        p_col = p_g.view(-1, 1)
        diff_b = b_col.ne(b_col.t())
        diff_p = p_col.ne(p_col.t())
        diag = torch.eye(M, dtype=torch.bool, device=dist_in.device)
        pos_mask_in = same_t & diff_b & diff_p & (~diag)
        neg_mask_in = (~same_t) & diff_b & diff_p & (~diag)

        # In-batch hardest positive and negatives
        pos_dist_in = dist_in.masked_fill(~pos_mask_in, float("-inf")).max(dim=1).values
        neg_dists_in = dist_in.masked_fill(~neg_mask_in, float("inf"))

        # Memory positives/negatives under NSBP
        pos_dist_mem = torch.full((M,), float("-inf"), device=X.device)
        neg_dists_mem_min = None
        if (
            bool(getattr(self, "triplet_use_memory", False))
            and getattr(self, "_trip_mem_feats", None) is not None
            and getattr(self, "_trip_mem_labels", None) is not None
            and getattr(self, "_trip_mem_batches", None) is not None
            and getattr(self, "_trip_mem_perts", None) is not None
            and self._trip_mem_labels.numel() > 0
        ):
            mem_feats = self._trip_mem_feats.detach().clone()
            mem_labels = self._trip_mem_labels.detach().clone()
            mem_batches = self._trip_mem_batches.detach().clone()
            mem_perts = self._trip_mem_perts.detach().clone()
            if metric == "cosine":
                dist_mem = (1.0 - (Xn @ mem_feats.t())).clamp_min(0.0)
            else:
                dist_mem = torch.cdist(Xn, mem_feats, p=2)
            y_row_g = y_g.view(-1, 1)
            b_row = b_g.view(-1, 1)
            p_row = p_g.view(-1, 1)
            mem_pos_mask = (
                y_row_g.eq(mem_labels.view(1, -1))
                & b_row.ne(mem_batches.view(1, -1))
                & p_row.ne(mem_perts.view(1, -1))
            )
            mem_neg_mask = (
                y_row_g.ne(mem_labels.view(1, -1))
                & b_row.ne(mem_batches.view(1, -1))
                & p_row.ne(mem_perts.view(1, -1))
            )
            pos_dist_mem = (
                dist_mem.masked_fill(~mem_pos_mask, float("-inf")).max(dim=1).values
            )
            neg_dists_mem_min = (
                dist_mem.masked_fill(~mem_neg_mask, float("inf")).min(dim=1).values
            )

        # Combine positives
        pos_dist = torch.maximum(pos_dist_in, pos_dist_mem)

        # Semi-hard mining: select closest negative with distance > pos_dist
        use_semi = bool(getattr(self, "triplet_semi_hard", True))
        if use_semi:
            semi_mask_in = neg_dists_in > pos_dist.view(-1, 1)
            semi_neg_in = (
                neg_dists_in.masked_fill(~semi_mask_in, float("inf")).min(dim=1).values
            )
            neg_candidates = [semi_neg_in]
            if neg_dists_mem_min is not None:
                semi_mem = torch.where(
                    neg_dists_mem_min > pos_dist,
                    neg_dists_mem_min,
                    torch.tensor(float("inf"), device=X.device),
                )
                neg_candidates.append(semi_mem)
            neg_dist = torch.stack(neg_candidates, dim=0).min(dim=0).values
            # Fallback to hardest negative if none semi-hard
            no_semi = torch.isinf(neg_dist)
            if no_semi.any():
                hard_in = neg_dists_in.min(dim=1).values
                if neg_dists_mem_min is not None:
                    hard_overall = torch.minimum(hard_in, neg_dists_mem_min)
                else:
                    hard_overall = hard_in
                neg_dist = torch.where(no_semi, hard_overall, neg_dist)
        else:
            hard_in = neg_dists_in.min(dim=1).values
            if neg_dists_mem_min is not None:
                neg_dist = torch.minimum(hard_in, neg_dists_mem_min)
            else:
                neg_dist = hard_in

        margin = float(getattr(self, "triplet_margin", 0.2))
        # Valid anchors: need any positive and any negative
        has_pos_any = torch.isfinite(pos_dist)
        has_neg_any = torch.isfinite(neg_dist)
        valid_anchor = has_pos_any & has_neg_any
        if valid_anchor.sum().item() == 0:
            loss = torch.tensor(0.0, device=features_first_view.device)
            self.log(mode + "_triplet_loss", loss, prog_bar=False)
            return loss
        triplet = F.relu(pos_dist - neg_dist + margin)
        triplet = triplet[valid_anchor]
        loss = triplet.mean()
        # Mining health logs
        try:
            active_rate = valid_anchor.float().mean()
            self.log(
                mode + "_triplet_active_pairs", active_rate, prog_bar=(mode == "train")
            )
            pos_per_anchor = pos_mask_in.float().sum(dim=1).mean()
            self.log(
                mode + "_triplet_pos_per_anchor_inbatch", pos_per_anchor, prog_bar=False
            )
        except Exception:
            pass
        self.log(mode + "_triplet_loss", loss, prog_bar=(mode == "train"))
        # Update memory with normalized feats and global ids
        if mode == "train" and bool(getattr(self, "triplet_use_memory", False)):
            try:
                self._update_triplet_memory(
                    Xn.detach(), y_g.detach(), b_g.detach(), p_g.detach()
                )
            except Exception:
                pass
        return loss

    def initialize_teacher(self):
        """Freeze a teacher from the current backbone and snapshot initial weights for L2-SP.

        Call this AFTER loading the pretrained checkpoint.
        """
        try:
            self.teacher_backbone = copy.deepcopy(self.backbone).to(self.device)
            self.teacher_backbone.eval()
            for p in self.teacher_backbone.parameters():
                p.requires_grad = False
            # Snapshot initial backbone weights for optional L2-SP
            self.backbone_init_state = {
                name: p.detach().clone().to(self.device)
                for name, p in self.backbone.named_parameters()
            }
            print("🧊 Initialized frozen teacher backbone and L2-SP anchors")
        except Exception as e:
            print(f"⚠️  Failed to initialize teacher/backbone anchors: {e}")

    # ---------------- Backbone unfreezing and LLRD helpers ----------------
    def _prepare_backbone_blocks(self):
        self._vit_blocks = []
        self._num_vit_blocks = 0
        self._patch_embed = getattr(self.backbone, "patch_embed", None)
        self._pos_embed = getattr(self.backbone, "pos_embed", None)
        self._cls_token = getattr(self.backbone, "cls_token", None)
        self._final_norm = getattr(self.backbone, "norm", None)
        if hasattr(self.backbone, "blocks") and isinstance(
            self.backbone.blocks, nn.ModuleList
        ):
            self._vit_blocks = list(self.backbone.blocks)
            self._num_vit_blocks = len(self._vit_blocks)

    def _set_trainable_blocks(self, num_trainable_blocks: int):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if not self.freeze_patch_embed:
            for mod in [self._patch_embed, self._pos_embed, self._cls_token]:
                if mod is None:
                    continue
                if isinstance(mod, nn.Parameter):
                    mod.requires_grad = True
                else:
                    for p in getattr(mod, "parameters", lambda: [])():
                        p.requires_grad = True
        k = max(0, min(int(num_trainable_blocks), self._num_vit_blocks))
        if k > 0:
            for blk in self._vit_blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
        if k > 0 and self._final_norm is not None:
            for p in self._final_norm.parameters():
                p.requires_grad = True
        self.current_trainable_blocks = k

    def _build_backbone_llrd_param_groups(
        self, base_lr: float, decay: float, wd: float
    ):
        param_groups = []

        def is_no_decay(n: str, p: torch.nn.Parameter):
            if p.ndim == 1:
                return True
            if "bias" in n:
                return True
            if "pos_embed" in n or "cls_token" in n:
                return True
            return False

        num_blocks = getattr(self, "_num_vit_blocks", 0)
        vit_blocks = getattr(self, "_vit_blocks", [])
        for i, blk in enumerate(vit_blocks):
            lr_scale = decay ** (num_blocks - 1 - i if num_blocks > 0 else 0)
            decay_params, no_decay_params = [], []
            for n, p in blk.named_parameters():
                (no_decay_params if is_no_decay(n, p) else decay_params).append(p)
            if decay_params:
                param_groups.append(
                    {
                        "name": f"backbone_block_{i}_decay",
                        "params": decay_params,
                        "lr": base_lr * lr_scale,
                        "weight_decay": wd,
                    }
                )
            if no_decay_params:
                param_groups.append(
                    {
                        "name": f"backbone_block_{i}_no_decay",
                        "params": no_decay_params,
                        "lr": base_lr * lr_scale,
                        "weight_decay": 0.0,
                    }
                )
        tail_lr = base_lr * (decay ** max(1, num_blocks)) if num_blocks > 0 else base_lr
        special_modules = []
        if getattr(self, "_patch_embed", None) is not None:
            special_modules.append(("patch_embed", self._patch_embed))
        if isinstance(getattr(self, "_pos_embed", None), torch.nn.Parameter):
            special_modules.append(("pos_embed", self._pos_embed))
        if isinstance(getattr(self, "_cls_token", None), torch.nn.Parameter):
            special_modules.append(("cls_token", self._cls_token))
        if getattr(self, "_final_norm", None) is not None:
            special_modules.append(("final_norm", self._final_norm))
        for name, mod in special_modules:
            decay_params, no_decay_params = [], []
            if isinstance(mod, torch.nn.Parameter):
                no_decay_params.append(mod)
            else:
                for n, p in mod.named_parameters(recurse=True):
                    (no_decay_params if is_no_decay(n, p) else decay_params).append(p)
            if decay_params:
                param_groups.append(
                    {
                        "name": f"backbone_{name}_decay",
                        "params": decay_params,
                        "lr": tail_lr,
                        "weight_decay": wd,
                    }
                )
            if no_decay_params:
                param_groups.append(
                    {
                        "name": f"backbone_{name}_no_decay",
                        "params": no_decay_params,
                        "lr": tail_lr,
                        "weight_decay": 0.0,
                    }
                )
        return param_groups

    def on_train_epoch_start(self):
        try:
            epoch = int(self.current_epoch)
            start = int(
                getattr(
                    self.hparams,
                    "unfreeze_start_epoch",
                    getattr(self, "unfreeze_start_epoch", 5),
                )
            )
            every = int(
                getattr(
                    self.hparams, "unfreeze_every", getattr(self, "unfreeze_every", 5)
                )
            )
            max_k = int(
                getattr(
                    self.hparams,
                    "max_trainable_blocks",
                    getattr(self, "max_trainable_blocks", 2),
                )
            )
            if epoch < start:
                desired = 0
            else:
                steps = 1 + (epoch - start) // max(1, every)
                desired = max(0, min(steps, max_k))
            if desired != getattr(self, "current_trainable_blocks", 0):
                self._set_trainable_blocks(desired)
                self.log(
                    "train_unfrozen_blocks",
                    float(self.current_trainable_blocks),
                    prog_bar=True,
                )
                print(
                    f"  🔓 Unfreezing: last {self.current_trainable_blocks} ViT blocks trainable"
                )
        except Exception as e:
            print(f"⚠️  Unfreezing schedule failed: {e}")

    # ---------------- Triplet memory helpers ----------------
    def _init_triplet_memory(self, feature_dim: int):
        device = self.device
        N = int(
            getattr(
                self.hparams,
                "triplet_memory_size",
                getattr(self, "triplet_memory_size", 4096),
            )
        )
        self._trip_mem_feats = torch.zeros(N, feature_dim, device=device)
        self._trip_mem_labels = torch.full((N,), -1, device=device, dtype=torch.long)
        self._trip_mem_batches = torch.full((N,), -1, device=device, dtype=torch.long)
        self._trip_mem_perts = torch.full((N,), -1, device=device, dtype=torch.long)
        self._trip_mem_ptr = 0

    def _update_triplet_memory(
        self,
        feats_norm: torch.Tensor,
        target_ids: torch.Tensor,
        batch_ids: torch.Tensor,
        pert_ids: torch.Tensor,
    ):
        if not bool(getattr(self, "triplet_use_memory", False)):
            return
        if (
            feats_norm is None
            or target_ids is None
            or batch_ids is None
            or pert_ids is None
        ):
            return
        if getattr(self, "_trip_mem_feats", None) is None:
            self._init_triplet_memory(feats_norm.shape[1])
        mask = (target_ids >= 0) & (batch_ids >= 0) & (pert_ids >= 0)
        if mask.sum() == 0:
            return
        feats = feats_norm[mask]
        labs = target_ids[mask]
        b_ids = batch_ids[mask]
        p_ids = pert_ids[mask]
        N = self._trip_mem_feats.shape[0]
        m = feats.shape[0]
        if m == 0:
            return
        end = self._trip_mem_ptr + m
        if end <= N:
            self._trip_mem_feats[self._trip_mem_ptr : end] = feats
            self._trip_mem_labels[self._trip_mem_ptr : end] = labs
            self._trip_mem_batches[self._trip_mem_ptr : end] = b_ids
            self._trip_mem_perts[self._trip_mem_ptr : end] = p_ids
        else:
            first = N - self._trip_mem_ptr
            if first > 0:
                self._trip_mem_feats[self._trip_mem_ptr : N] = feats[:first]
                self._trip_mem_labels[self._trip_mem_ptr : N] = labs[:first]
                self._trip_mem_batches[self._trip_mem_ptr : N] = b_ids[:first]
                self._trip_mem_perts[self._trip_mem_ptr : N] = p_ids[:first]
            rem = m - first
            self._trip_mem_feats[0:rem] = feats[first:]
            self._trip_mem_labels[0:rem] = labs[first:]
            self._trip_mem_batches[0:rem] = b_ids[first:]
            self._trip_mem_perts[0:rem] = p_ids[first:]
        self._trip_mem_ptr = (self._trip_mem_ptr + m) % N

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

        # Add backbone groups with LLRD
        backbone_groups = self._build_backbone_llrd_param_groups(
            base_lr=projector_lr
            * float(
                getattr(
                    self.hparams,
                    "backbone_lr_scale",
                    getattr(self, "backbone_lr_scale", 0.2),
                )
            ),
            decay=float(
                getattr(self.hparams, "llrd_decay", getattr(self, "llrd_decay", 0.65))
            ),
            wd=self.hparams.weight_decay,
        )

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
            + backbone_groups
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
                    total = domain_dist.sum()
                    if total > 0:
                        probs = domain_dist / total
                        entropy = -np.sum(probs * np.log(probs + 1e-8))
                        self.log(
                            f"{mode}_domain_entropy", float(entropy), prog_bar=False
                        )

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

        imgs, metadata_collated, domain_labels = batch
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

        # Anti-forgetting regularizers (Feature KD, Relational KD, Adapter anchor, L2-SP)
        kd_feat_loss = torch.tensor(0.0, device=self.device)
        rel_kd_loss = torch.tensor(0.0, device=self.device)
        adapter_anchor_loss = torch.tensor(0.0, device=self.device)
        l2sp_loss = torch.tensor(0.0, device=self.device)

        # Feature KD and Relational KD against frozen teacher
        if getattr(self, "teacher_backbone", None) is not None and (
            self.distill_weight > 0.0 or self.rel_kd_weight > 0.0
        ):
            with torch.no_grad():
                teacher_feats = self.teacher_backbone(imgs)
            # Feature KD: match adapted export to teacher backbone features
            if self.distill_weight > 0.0:
                kd_feat_loss = F.mse_loss(adapted, teacher_feats)
                self.log("train_kd_feat_loss", kd_feat_loss, prog_bar=False)
            # Relational KD: match cosine similarity structure
            if self.rel_kd_weight > 0.0:
                s_student = (
                    F.normalize(adapted, dim=1) @ F.normalize(adapted, dim=1).t()
                )
                s_teacher = (
                    F.normalize(teacher_feats, dim=1)
                    @ F.normalize(teacher_feats, dim=1).t()
                )
                rel_kd_loss = F.mse_loss(s_student, s_teacher)
                self.log("train_rel_kd_loss", rel_kd_loss, prog_bar=False)

        # Adapter anchor: keep adapter delta small
        if self.adapter_anchor_weight > 0.0:
            with torch.no_grad():
                base_feats = self.backbone(imgs)
            adapter_delta = adapted - base_feats
            adapter_anchor_loss = adapter_delta.pow(2).mean()
            self.log("train_adapter_anchor_loss", adapter_anchor_loss, prog_bar=False)

        # L2-SP: only if any backbone params are trainable
        if self.l2sp_weight > 0.0 and any(
            p.requires_grad for p in self.backbone.parameters()
        ):
            try:
                accum = 0.0
                count = 0
                for name, p in self.backbone.named_parameters():
                    if not p.requires_grad:
                        continue
                    if (
                        self.backbone_init_state is None
                        or name not in self.backbone_init_state
                    ):
                        continue
                    p0 = self.backbone_init_state[name]
                    accum = accum + (p - p0).pow(2).mean()
                    count += 1
                if count > 0:
                    l2sp_loss = accum / float(count)
                    self.log("train_l2sp_loss", l2sp_loss, prog_bar=False)
            except Exception as e:
                print(f"⚠️  L2-SP computation failed: {e}")

        # Triplet loss on first-view adapted embeddings
        triplet_loss = torch.tensor(0.0, device=self.device)
        try:
            Bfirst = imgs.shape[0] // 2
            first_view_adapted = adapted[:Bfirst]
            triplet_loss = self._batch_hard_triplet_loss(
                first_view_adapted, metadata_collated, mode="train"
            )
        except Exception as e:
            if batch_idx % 10 == 0:
                print(f"⚠️  Triplet loss (train) failed: {e}")

        # Combine losses
        total_loss = (
            simclr_loss
            + self.domain_loss_weight * domain_loss
            + self.triplet_weight * triplet_loss
            + self.distill_weight * kd_feat_loss
            + self.rel_kd_weight * rel_kd_loss
            + self.l2sp_weight * l2sp_loss
            + self.adapter_anchor_weight * adapter_anchor_loss
        )
        self.log("train_total_loss", total_loss, prog_bar=True)
        if float(getattr(self, "triplet_weight", 0.0)) > 0.0:
            self.log(
                "train_triplet_loss_weighted",
                self.triplet_weight * triplet_loss,
                prog_bar=False,
            )

        # Log GRL lambda for tracking
        self.log("train_grl_lambda", self.grl.lambd, prog_bar=False)

        # Log detailed progress every 10 batches
        if batch_idx % 10 == 0:
            print(
                f"    [LOSSES] SimCLR: {simclr_loss:.4f}, Weighted Triplet: {self.triplet_weight * triplet_loss:.4f}, Weighted Domain: {self.domain_loss_weight * domain_loss:.4f}, Total: {total_loss:.4f}"
            )
            print(f"    [GRL] Current Lambda: {self.grl.lambd:.3f}")

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Progress tracking for validation
        if batch_idx % 2 == 0:  # Print every 2 validation batches
            print(f"  [VAL] Epoch {self.current_epoch}, Batch {batch_idx}")

        imgs, metadata_collated, domain_labels = batch
        imgs = torch.cat(imgs, dim=0)
        feats, penultimate, adapted = self._forward_embeddings(imgs)
        _ = self._simclr_loss_and_metrics(feats, mode="val")
        domain_targets = (
            domain_labels.detach().clone().to(device=self.device, dtype=torch.long)
        )
        domain_targets = torch.cat([domain_targets, domain_targets], dim=0)
        _ = self._domain_loss_and_metrics(adapted, domain_targets, mode="val")
        # Triplet loss on first-view embeddings only (for logging)
        try:
            Bfirst = imgs.shape[0] // 2
            first_view_adapted = adapted[:Bfirst]
            triplet_loss_val = self._batch_hard_triplet_loss(
                first_view_adapted, metadata_collated, mode="val"
            )
            if float(getattr(self, "triplet_weight", 0.0)) > 0.0:
                self.log(
                    "val_triplet_loss_weighted",
                    self.triplet_weight * triplet_loss_val,
                    prog_bar=False,
                )
        except Exception as e:
            if batch_idx % 10 == 0:
                print(f"⚠️  Triplet loss (val) failed: {e}")

    def _mini_eval_and_log(self):
        """Run a validation-style eval over the entire val loader and log metrics."""
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
            adv_label = str(getattr(self.hparams, "domain_label_key", "batch"))
            print(
                f"🔎 Label alignment: adversary label='{adv_label}'; leakage logged for ['batch','source','plate'] if present"
            )

            self.eval()
            device = self.device
            rows = []
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

                    # continue through entire val_loader for more stable mini-eval

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

            # Leakage proxies for multiple labels using same embeddings
            batch_auc_val = None
            for label_key in ["batch", "source", "plate"]:
                try:
                    if label_key not in df.columns:
                        continue
                    if df.get(label_key).nunique(dropna=True) < 2:
                        print(f"⚠️  mini-eval: only one class for {label_key}; skipping")
                        continue
                    df_for_label = df.copy()
                    # Re-map to 'batch' column expected by evl.batch_classification
                    df_for_label["batch"] = df_for_label[label_key].astype(str)
                    metrics = evl.batch_classification(df_for_label)
                    self.log(
                        f"val_leak_{label_key}_MCC",
                        float(metrics.get("MCC", float("nan"))),
                    )
                    auc_val = float(metrics.get("ROC_AUC", float("nan")))
                    self.log(f"val_leak_{label_key}_ROC_AUC", auc_val)
                    if label_key == str(
                        getattr(self.hparams, "domain_label_key", "batch")
                    ):
                        batch_auc_val = auc_val
                except Exception as e:
                    print(f"⚠️  leakage eval failed for {label_key}: {e}")

            # Optional: XGBoost-based masking within mini-eval for reporting
            try:
                frac = float(
                    getattr(
                        self.hparams,
                        "xgb_mask_fraction",
                        getattr(self, "xgb_mask_fraction", 0.0),
                    )
                )
                if frac > 0.0 and len(rows) >= 20:
                    try:
                        import xgboost as xgb
                    except Exception as e:
                        print(f"⚠️  XGBoost unavailable for mini-eval masking: {e}")
                        raise RuntimeError

                    df_xgb = df.copy()
                    emb_cols = [c for c in df_xgb.columns if c.startswith("emb")]
                    if (
                        len(emb_cols) >= 2
                        and df_xgb.get("batch").nunique(dropna=True) >= 2
                    ):
                        X = df_xgb[emb_cols].to_numpy()
                        y_raw = df_xgb["batch"].astype(str).fillna("unknown").values
                        # Encode labels
                        from sklearn.preprocessing import LabelEncoder

                        le = LabelEncoder()
                        y = le.fit_transform(y_raw)
                        clf = xgb.XGBClassifier(
                            n_estimators=400,
                            max_depth=5,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            random_state=int(
                                getattr(
                                    self.hparams,
                                    "xgb_random_state",
                                    getattr(self, "xgb_random_state", 42),
                                )
                            ),
                            n_jobs=4,
                        )
                        clf.fit(X, y)
                        booster = clf.get_booster()
                        try:
                            score_dict = booster.get_score(importance_type="gain")
                        except Exception:
                            score_dict = booster.get_score(importance_type="weight")
                        importances = np.zeros(len(emb_cols), dtype=float)
                        for k, v in score_dict.items():
                            try:
                                idx = int(k[1:])
                                if 0 <= idx < len(importances):
                                    importances[idx] = float(v)
                            except Exception:
                                continue
                        # Mask top fraction
                        n_drop = int(np.floor(len(emb_cols) * frac))
                        n_drop = max(0, min(n_drop, len(emb_cols)))
                        order = np.argsort(-importances)
                        drop_cols = [emb_cols[i] for i in order[:n_drop]]
                        keep_cols = [c for c in emb_cols if c not in drop_cols]

                        # Recompute post-mask metrics on df_xgb with kept features only
                        df_mask = pd.concat(
                            [
                                df_xgb[
                                    [
                                        c
                                        for c in df_xgb.columns
                                        if not c.startswith("emb")
                                    ]
                                ],
                                df_xgb[keep_cols],
                            ],
                            axis=1,
                        )

                        # Target NSP/NSBP post-mask
                        try:
                            if (
                                "target" in df_mask.columns
                                and df_mask["target"].notnull().any()
                            ):
                                feature_cols_m = [
                                    c for c in df_mask.columns if c.startswith("emb")
                                ]
                                X_t = StandardScaler().fit_transform(
                                    df_mask[feature_cols_m].to_numpy()
                                )
                                y_t = LabelEncoder().fit_transform(
                                    df_mask["target"].astype(str).values
                                )
                                wells = (
                                    df_mask["well"].astype(str).values
                                    if "well" in df_mask.columns
                                    else np.array([""] * len(df_mask))
                                )
                                batches_t = (
                                    df_mask["batch"].astype(str).values
                                    if "batch" in df_mask.columns
                                    else np.array([""] * len(df_mask))
                                )
                                ypred_nsp = evl.nearest_neighbor_classifier_NSBW(
                                    X_t, y_t, mode="NSW", wells=wells, metric="cosine"
                                )
                                self.log(
                                    "val_target_acc_NSP_postmask",
                                    float((ypred_nsp == y_t).mean()),
                                )
                                ypred_nsbp = evl.nearest_neighbor_classifier_NSBW(
                                    X_t,
                                    y_t,
                                    mode="NSBW",
                                    batches=batches_t,
                                    wells=wells,
                                    metric="cosine",
                                )
                                self.log(
                                    "val_target_acc_NSBP_postmask",
                                    float((ypred_nsbp == y_t).mean()),
                                )
                        except Exception as e:
                            print(f"⚠️  post-mask target metrics failed: {e}")

                        # Log drop stats
                        try:
                            self.log("val_xgb_mask_drop_fraction", float(frac))
                            self.log("val_xgb_mask_drop_count", float(len(drop_cols)))
                        except Exception:
                            pass
                    else:
                        print(
                            "⚠️  XGB mini-eval masking skipped: insufficient emb cols or batches"
                        )
            except Exception:
                pass

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

            # Target accuracies: NSP (NSW) and NSBP (NSBW)
            try:
                if "target" in df.columns and df["target"].notnull().any():
                    X_t = StandardScaler().fit_transform(df[feature_cols].to_numpy())
                    y_t = LabelEncoder().fit_transform(df["target"].astype(str).values)
                    wells = (
                        df["well"].astype(str).values
                        if "well" in df.columns
                        else np.array([""] * len(df))
                    )
                    batches_t = (
                        df["batch"].astype(str).values
                        if "batch" in df.columns
                        else np.array([""] * len(df))
                    )
                    # NSP ~ NSW in code: exclude same well
                    ypred_nsp = evl.nearest_neighbor_classifier_NSBW(
                        X_t, y_t, mode="NSW", wells=wells, metric="cosine"
                    )
                    nsp_acc = (ypred_nsp == y_t).mean()
                    self.log("val_target_acc_NSP", float(nsp_acc))
                    # NSBP ~ NSBW in code: exclude same batch and same well
                    ypred_nsbp = evl.nearest_neighbor_classifier_NSBW(
                        X_t,
                        y_t,
                        mode="NSBW",
                        batches=batches_t,
                        wells=wells,
                        metric="cosine",
                    )
                    nsbp_acc = (ypred_nsbp == y_t).mean()
                    self.log("val_target_acc_NSBP", float(nsbp_acc))
                else:
                    print(
                        "⚠️  mini-eval: 'target' not available for target accuracy logging"
                    )
            except Exception as e:
                print(f"⚠️  Target accuracy eval failed: {e}")

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

            # Target NSBP metrics: OR@5, hit rate@5, eligible rate
            try:
                if "target" in df.columns and df["target"].notnull().any():
                    targets_arr = df["target"].astype(str).values
                    batches_arr = (
                        df["batch"].astype(str).values
                        if "batch" in df.columns
                        else None
                    )
                    perts_arr = (
                        df["perturbation_id"].astype(str).values
                        if "perturbation_id" in df.columns
                        else None
                    )
                    if batches_arr is None or perts_arr is None:
                        print(
                            "⚠️  mini-eval: missing batch or perturbation_id for NSBP; skipping"
                        )
                    else:
                        # Normalize features and compute similarities on the fly
                        E = df[feature_cols].to_numpy().astype(np.float32)
                        # row-normalize
                        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
                        En = E / norms
                        K = 5
                        a_tot = 0
                        b_tot = 0
                        c_tot = 0
                        d_tot = 0
                        eligible_k = 0
                        hits = 0
                        N = En.shape[0]
                        for i in range(N):
                            cand = (batches_arr != batches_arr[i]) & (
                                perts_arr != perts_arr[i]
                            )
                            cand[i] = False
                            idxs = np.where(cand)[0]
                            if idxs.size == 0:
                                continue
                            # cosine similarities to candidates
                            sims = En[idxs] @ En[i]
                            topk = min(K, idxs.size)
                            if topk == 0:
                                continue
                            # eligibility: at least one NSBP same-target exists in pool
                            same_cand = targets_arr[idxs] == targets_arr[i]
                            same_total = int(np.sum(same_cand))
                            if same_total > 0:
                                eligible_k += 1
                            # select topK indices
                            part = np.argpartition(sims, -topk)[-topk:]
                            top_idx = idxs[part]
                            # count positives among candidates and among topK
                            same_top = targets_arr[top_idx] == targets_arr[i]
                            a_i = int(np.sum(same_top))
                            c_i = topk - a_i
                            b_i = max(0, same_total - a_i)
                            d_i = max(0, int(idxs.size) - topk - b_i)
                            a_tot += a_i
                            b_tot += b_i
                            c_tot += c_i
                            d_tot += d_i
                            if same_total > 0 and a_i > 0:
                                hits += 1
                        # Compute pooled OR with Haldane-Anscombe correction
                        aa, bb, cc, dd = a_tot, b_tot, c_tot, d_tot
                        if min(aa, bb, cc, dd) == 0:
                            aa += 0.5
                            bb += 0.5
                            cc += 0.5
                            dd += 0.5
                        or_at5 = (aa / bb) / (cc / dd)
                        total_queries = N
                        eligible_rate = eligible_k / max(1, total_queries)
                        hit_rate = hits / max(1, eligible_k)
                        self.log("val_target_OR_at5_NSBP", float(or_at5))
                        self.log("val_target_at5_NSBP", float(hit_rate))
                        self.log("val_target_eligible_rate_NSBP", float(eligible_rate))
                else:
                    print("⚠️  mini-eval: 'target' column missing for NSBP metrics")
            except Exception as e:
                print(f"⚠️  Target NSBP metrics eval failed: {e}")

            # Early-stop checkpointing using target accuracy (NSP, NSBP) as primary criteria
            try:
                # Pull the most recently logged target accuracies from LoggerCollection
                # We compute both NSP and NSBP; prefer NSBP, fallback to NSP.
                nsp = self.trainer.callback_metrics.get("val_target_acc_NSP", None)
                nsbp = self.trainer.callback_metrics.get("val_target_acc_NSBP", None)
                target_metric = None
                if nsbp is not None and not torch.isnan(nsbp):
                    target_metric = float(nsbp.detach().cpu().item())
                    target_name = "NSBP"
                elif nsp is not None and not torch.isnan(nsp):
                    target_metric = float(nsp.detach().cpu().item())
                    target_name = "NSP"

                if target_metric is not None:
                    # Keep best target accuracy, break ties by lower leakage AUC if available
                    best_key = f"best_target_acc_{target_name.lower()}"
                    current_best = getattr(self, best_key, -1.0)
                    leakage = (
                        batch_auc_val
                        if (batch_auc_val is not None and not np.isnan(batch_auc_val))
                        else None
                    )

                    better_target = target_metric > (current_best + 1e-6)
                    tie_break = (
                        (not better_target)
                        and abs(target_metric - current_best) <= 1e-6
                        and (
                            leakage is not None
                            and leakage < (self.best_batch_auc - 1e-6)
                        )
                    )
                    if better_target or tie_break:
                        setattr(self, best_key, target_metric)
                        if leakage is not None:
                            self.best_batch_auc = leakage
                        ckpt_dir = self.trainer.default_root_dir or "."
                        fname = f"earlystop_epoch{self.current_epoch:02d}_target{target_name}_{target_metric:.4f}.ckpt"
                        path = os.path.join(ckpt_dir, fname)
                        self.trainer.save_checkpoint(path)
                        print(
                            f"💾 Saved early-stop checkpoint on best target {target_name}: {path}"
                        )
                        # Log best-so-far values
                        self.log(f"val_best_target_acc_{target_name}", target_metric)
                        if leakage is not None:
                            self.log("val_best_batch_ROC_AUC", self.best_batch_auc)
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
        default=0.2,
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
        default=1.5,
        help="Domain head LR as a scale of projector_lr (e.g., 1.5 means head is 1.5x)",
    )
    parser.add_argument(
        "--head_update_every",
        type=int,
        default=1,
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
    parser.add_argument(
        "--domain_label",
        type=str,
        default="batch",
        choices=["batch", "source", "plate"],
        help="Which metadata label to train the adversary on",
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
    # Anti-forgetting weights
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=0.5,
        help="Weight for feature knowledge distillation (student adapted vs teacher backbone)",
    )
    parser.add_argument(
        "--rel_kd_weight",
        type=float,
        default=0.1,
        help="Weight for relational KD (match cosine-similarity matrices)",
    )
    parser.add_argument(
        "--l2sp_weight",
        type=float,
        default=1e-3,
        help="Weight for L2-SP on any unfrozen backbone params",
    )
    parser.add_argument(
        "--adapter_anchor_weight",
        type=float,
        default=1e-4,
        help="Weight for adapter anchor penalty (keep adapter delta small)",
    )

    # Triplet loss (target-separating) parameters
    parser.add_argument(
        "--triplet_weight",
        type=float,
        default=0.5,
        help="Weight for batch-hard triplet loss (0 disables)",
    )
    parser.add_argument(
        "--triplet_margin",
        type=float,
        default=0.1,
        help="Margin for triplet loss",
    )
    parser.add_argument(
        "--triplet_metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for triplet loss",
    )
    # Progressive unfreezing & LLRD
    parser.add_argument(
        "--unfreeze_start_epoch",
        type=int,
        default=5,
        help="Epoch to begin unfreezing ViT blocks",
    )
    parser.add_argument(
        "--unfreeze_every",
        type=int,
        default=5,
        help="Unfreeze +1 block every N epochs after start",
    )
    parser.add_argument(
        "--max_trainable_blocks",
        type=int,
        default=2,
        help="Maximum number of last ViT blocks to unfreeze",
    )
    parser.add_argument(
        "--llrd_decay",
        type=float,
        default=0.65,
        help="Layer-wise LR decay factor (0<d<1)",
    )
    parser.add_argument(
        "--backbone_lr_scale",
        type=float,
        default=0.2,
        help="Backbone base LR = projector_lr * scale",
    )
    parser.add_argument(
        "--no_freeze_patch_embed",
        action="store_true",
        help="Allow patch/embed tokens to train when unfreezing",
    )
    # Triplet refinements
    parser.add_argument(
        "--triplet_use_memory",
        action="store_true",
        help="Enable cross-batch memory for negatives",
    )
    parser.add_argument(
        "--triplet_memory_size",
        type=int,
        default=16384,
        help="Size of cross-batch memory queue",
    )
    parser.add_argument(
        "--no_triplet_semi_hard",
        action="store_true",
        help="Disable semi-hard mining (use hardest)",
    )
    # XGBoost-based feature masking in validation reporting
    parser.add_argument(
        "--xgb_mask_fraction",
        type=float,
        default=0.0,
        help="If >0, compute XGBoost batch importances in mini-eval, drop top fraction, and log post-mask metrics.",
    )
    parser.add_argument(
        "--xgb_top_k_plot",
        type=int,
        default=40,
        help="Top-K feature importances to plot during validation mini-eval.",
    )
    parser.add_argument(
        "--xgb_random_state",
        type=int,
        default=42,
        help="Random state for XGBoost in validation mini-eval.",
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
    domain_label_key = args.domain_label
    warmup_epochs = args.warmup_epochs
    lr_final_value = args.lr_final_value
    adapter_hidden = args.adapter_hidden
    adapter_scale = args.adapter_scale
    domain_loss_weight = args.domain_loss_weight
    # Anti-forgetting
    distill_weight = args.distill_weight
    rel_kd_weight = args.rel_kd_weight
    l2sp_weight = args.l2sp_weight
    adapter_anchor_weight = args.adapter_anchor_weight
    # Triplet loss
    triplet_weight = args.triplet_weight
    triplet_margin = args.triplet_margin
    triplet_metric = args.triplet_metric
    # Progressive unfreezing & LLRD
    unfreeze_start_epoch = args.unfreeze_start_epoch
    unfreeze_every = args.unfreeze_every
    max_trainable_blocks = args.max_trainable_blocks
    llrd_decay = args.llrd_decay
    backbone_lr_scale = args.backbone_lr_scale
    freeze_patch_embed = not args.no_freeze_patch_embed
    # Triplet refinements
    triplet_use_memory = args.triplet_use_memory
    triplet_memory_size = args.triplet_memory_size
    triplet_semi_hard = not args.no_triplet_semi_hard

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
        domain_label_key=domain_label_key,
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
    print(f"🔎 Adversary domain label column: '{domain_label_key}'")
    print("🔎 Leakage will be logged for: ['batch','source','plate'] if present")

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
                "domain_loss_weight": domain_loss_weight,
                "domain_hidden": domain_hidden,
                "domain_head_dropout": args.domain_head_dropout,
                "freeze_encoder": freeze_encoder,
                "balanced_batches": balanced_batches,
                "domain_label_key": domain_label_key,
                "num_workers": num_workers,
                "train_ratio": train_ratio,
                "num_domains": num_domains,
                "adapter_hidden": adapter_hidden,
                "adapter_scale": adapter_scale,
                "projector_lr": args.projector_lr,
                "head_lr_scale": args.head_lr_scale,
                "head_update_every": args.head_update_every,
                "distill_weight": distill_weight,
                "rel_kd_weight": rel_kd_weight,
                "l2sp_weight": l2sp_weight,
                "adapter_anchor_weight": adapter_anchor_weight,
                "triplet_weight": triplet_weight,
                "triplet_margin": triplet_margin,
                "triplet_metric": triplet_metric,
                # progressive unfreezing & LLRD
                "unfreeze_start_epoch": unfreeze_start_epoch,
                "unfreeze_every": unfreeze_every,
                "max_trainable_blocks": max_trainable_blocks,
                "llrd_decay": llrd_decay,
                "backbone_lr_scale": backbone_lr_scale,
                "freeze_patch_embed": freeze_patch_embed,
                # triplet refinements
                "triplet_use_memory": triplet_use_memory,
                "triplet_memory_size": triplet_memory_size,
                "triplet_semi_hard": triplet_semi_hard,
                # xgb masking params
                "xgb_mask_fraction": args.xgb_mask_fraction,
                "xgb_top_k_plot": args.xgb_top_k_plot,
                "xgb_random_state": args.xgb_random_state,
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
                save_top_k=-1,  # Save all checkpoints
                monitor=None,
                every_n_epochs=every_n_epochs,  # Save every n epochs from args
                save_last=True,  # Always save the last checkpoint
                filename="epoch_{epoch:02d}",
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
        domain_label_key=domain_label_key,
        distill_weight=distill_weight,
        rel_kd_weight=rel_kd_weight,
        l2sp_weight=l2sp_weight,
        adapter_anchor_weight=adapter_anchor_weight,
        triplet_weight=triplet_weight,
        triplet_margin=triplet_margin,
        triplet_metric=triplet_metric,
        # unfreezing & LLRD
        unfreeze_start_epoch=unfreeze_start_epoch,
        unfreeze_every=unfreeze_every,
        max_trainable_blocks=max_trainable_blocks,
        llrd_decay=llrd_decay,
        backbone_lr_scale=backbone_lr_scale,
        freeze_patch_embed=freeze_patch_embed,
        # triplet refinements
        triplet_use_memory=triplet_use_memory,
        triplet_memory_size=triplet_memory_size,
        triplet_semi_hard=triplet_semi_hard,
        # xgb masking params
        xgb_mask_fraction=args.xgb_mask_fraction,
        xgb_top_k_plot=args.xgb_top_k_plot,
        xgb_random_state=args.xgb_random_state,
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

    # Initialize frozen teacher and L2-SP anchors from the just-loaded backbone
    try:
        model.initialize_teacher()
    except Exception as e:
        print(f"⚠️  initialize_teacher failed: {e}")

    # ------------------------------------------------------------------
    # Baseline (pretrained) validation logging BEFORE training starts
    # Uses backbone features only (no adapter) to reflect pretrained baseline
    # ------------------------------------------------------------------
    try:
        print("🧪 Running baseline validation (pretrained backbone) before training...")
        prev_mode = model.training
        model.eval()
        device = model.device
        rows = []
        with torch.no_grad():
            for batch in val_loader:
                try:
                    imgs, metadata_collated, _domain = batch
                except Exception:
                    try:
                        imgs, metadata_collated = batch
                    except Exception:
                        break

                # use first view only for embedding consistency
                view = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
                view = view.to(device)

                backbone_feats = model.backbone(view)
                embs = backbone_feats.detach().cpu().numpy()

                B = embs.shape[0]
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
                        "perturbation_id": meta.get("compound", None),
                        "target": meta.get("target", None),
                    }
                    for j, v in enumerate(embs[i]):
                        row[f"emb{j}"] = float(v)
                    rows.append(row)

        baseline_metrics = {}
        if len(rows) >= 20:
            df = pd.DataFrame(rows)
            feature_cols = [c for c in df.columns if c.startswith("emb")]

            # Leakage proxies for multiple labels
            for label_key in ["batch", "source", "plate"]:
                try:
                    if label_key not in df.columns:
                        continue
                    if df.get(label_key).nunique(dropna=True) < 2:
                        continue
                    df_for_label = df.copy()
                    df_for_label["batch"] = df_for_label[label_key].astype(str)
                    metrics = evl.batch_classification(df_for_label)
                    baseline_metrics[f"baseline_leak_{label_key}_MCC"] = float(
                        metrics.get("MCC", float("nan"))
                    )
                    baseline_metrics[f"baseline_leak_{label_key}_ROC_AUC"] = float(
                        metrics.get("ROC_AUC", float("nan"))
                    )
                except Exception:
                    pass

            # NSB perturbation accuracy
            try:
                if (
                    "perturbation_id" in df.columns
                    and df["perturbation_id"].notnull().any()
                ):
                    X = StandardScaler().fit_transform(df[feature_cols].to_numpy())
                    y = LabelEncoder().fit_transform(
                        df["perturbation_id"].astype(str).values
                    )
                    batches_base = (
                        df["batch"].astype(str).values
                        if "batch" in df.columns
                        else np.array([""] * len(df))
                    )
                    if len(np.unique(batches_base)) >= 2:
                        ypred_nsb = evl.nearest_neighbor_classifier_NSBW(
                            X, y, mode="NSB", batches=batches_base, metric="cosine"
                        )
                        baseline_metrics["baseline_pert_acc_NSB"] = float(
                            (ypred_nsb == y).mean()
                        )
            except Exception:
                pass

            # Target accuracies: NSP/NSBP
            try:
                if "target" in df.columns and df["target"].notnull().any():
                    X_t = StandardScaler().fit_transform(df[feature_cols].to_numpy())
                    y_t = LabelEncoder().fit_transform(df["target"].astype(str).values)
                    wells = (
                        df["well"].astype(str).values
                        if "well" in df.columns
                        else np.array([""] * len(df))
                    )
                    batches_t = (
                        df["batch"].astype(str).values
                        if "batch" in df.columns
                        else np.array([""] * len(df))
                    )
                    ypred_nsp = evl.nearest_neighbor_classifier_NSBW(
                        X_t, y_t, mode="NSW", wells=wells, metric="cosine"
                    )
                    baseline_metrics["baseline_target_acc_NSP"] = float(
                        (ypred_nsp == y_t).mean()
                    )
                    ypred_nsbp = evl.nearest_neighbor_classifier_NSBW(
                        X_t,
                        y_t,
                        mode="NSBW",
                        batches=batches_t,
                        wells=wells,
                        metric="cosine",
                    )
                    baseline_metrics["baseline_target_acc_NSBP"] = float(
                        (ypred_nsbp == y_t).mean()
                    )
            except Exception:
                pass

            # Perturbation mAP (aggregated)
            try:
                df_valid = df[
                    df["perturbation_id"].notnull()
                    & (df["perturbation_id"].astype(str) != "")
                ].copy()
                vc = df_valid["perturbation_id"].value_counts()
                keep_ids = set(vc[vc >= 2].index.tolist())
                df_valid = df_valid[df_valid["perturbation_id"].isin(keep_ids)].copy()
                if df_valid["perturbation_id"].nunique() >= 2:
                    agg = (
                        df_valid.groupby(["perturbation_id"])
                        .mean(numeric_only=True)
                        .reset_index()
                    )
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
                            baseline_metrics["baseline_pert_mAP"] = mAP_val
            except Exception:
                pass

            # Target odds ratio / p-value at top 5% similarity
            try:
                if "target" in df.columns and df["target"].notnull().any():
                    if df["target"].nunique(dropna=True) >= 2 and len(df) >= 30:
                        X = StandardScaler().fit_transform(df[feature_cols].to_numpy())
                        labels = LabelEncoder().fit_transform(
                            df["target"].astype(str).values
                        )
                        dist_mat = evl.pairwise_distances_parallel(X, metric="cosine")
                        np.fill_diagonal(dist_mat, np.nan)
                        sim = 1.0 - dist_mat
                        n = sim.shape[0]
                        iu, ju = np.triu_indices(n, k=1)
                        sim_vals = sim[iu, ju]
                        thresh = np.nanpercentile(sim_vals, 95.0)
                        high = sim_vals >= thresh
                        same = labels[iu] == labels[ju]
                        a = int(np.nansum(np.logical_and(high, same)))
                        c = int(np.nansum(np.logical_and(high, ~same)))
                        b = int(np.nansum(np.logical_and(~high, same)))
                        d = int(np.nansum(np.logical_and(~high, ~same)))
                        aa, bb, cc, dd = a, b, c, d
                        if min(aa, bb, cc, dd) == 0:
                            aa += 0.5
                            bb += 0.5
                            cc += 0.5
                            dd += 0.5
                        odds_ratio = (aa / bb) / (cc / dd)

                        def _logcomb(n, k):
                            return (
                                math.lgamma(n + 1)
                                - math.lgamma(k + 1)
                                - math.lgamma(n - k + 1)
                            )

                        logp = (
                            _logcomb(a + b, a)
                            + _logcomb(c + d, c)
                            - _logcomb(a + b + c + d, a + c)
                        )
                        p_value = float(np.exp(logp))
                        baseline_metrics["baseline_target_odds_ratio_top5"] = float(
                            odds_ratio
                        )
                        baseline_metrics["baseline_target_p_value_top5"] = float(
                            p_value
                        )
            except Exception:
                pass

            # Baseline NSBP metrics: OR@5, hit rate@5, eligible rate
            try:
                if (
                    "target" in df.columns
                    and df["target"].notnull().any()
                    and "batch" in df.columns
                    and "perturbation_id" in df.columns
                ):
                    targets_arr = df["target"].astype(str).values
                    batches_arr = df["batch"].astype(str).values
                    perts_arr = df["perturbation_id"].astype(str).values
                    E = df[feature_cols].to_numpy().astype(np.float32)
                    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
                    En = E / norms
                    K = 5
                    a_tot = b_tot = c_tot = d_tot = 0
                    eligible_k = 0
                    hits = 0
                    N = En.shape[0]
                    for i in range(N):
                        cand = (batches_arr != batches_arr[i]) & (
                            perts_arr != perts_arr[i]
                        )
                        cand[i] = False
                        idxs = np.where(cand)[0]
                        if idxs.size == 0:
                            continue
                        sims = En[idxs] @ En[i]
                        topk = min(K, idxs.size)
                        if topk == 0:
                            continue
                        if idxs.size >= K:
                            eligible_k += 1
                        part = np.argpartition(sims, -topk)[-topk:]
                        top_idx = idxs[part]
                        same_cand = targets_arr[idxs] == targets_arr[i]
                        same_top = targets_arr[top_idx] == targets_arr[i]
                        a_i = int(np.sum(same_top))
                        c_i = topk - a_i
                        same_total = int(np.sum(same_cand))
                        b_i = max(0, same_total - a_i)
                        d_i = max(0, int(idxs.size) - topk - b_i)
                        a_tot += a_i
                        b_tot += b_i
                        c_tot += c_i
                        d_tot += d_i
                        if a_i > 0:
                            hits += 1
                    aa, bb, cc, dd = a_tot, b_tot, c_tot, d_tot
                    if min(aa, bb, cc, dd) == 0:
                        aa += 0.5
                        bb += 0.5
                        cc += 0.5
                        dd += 0.5
                    or_at5 = (aa / bb) / (cc / dd)
                    total_queries = N
                    eligible_rate = eligible_k / max(1, total_queries)
                    hit_rate = hits / max(1, eligible_k)
                    baseline_metrics["baseline_target_OR_at5_NSBP"] = float(or_at5)
                    baseline_metrics["baseline_target_at5_NSBP"] = float(hit_rate)
                    baseline_metrics["baseline_target_eligible_rate_NSBP"] = float(
                        eligible_rate
                    )
            except Exception:
                pass

        # Log to wandb (if enabled)
        if args.use_wandb and baseline_metrics:
            try:
                wandb_logger.experiment.log(baseline_metrics, step=0)
                print(
                    f"✅ Baseline metrics logged to wandb: {list(baseline_metrics.keys())}"
                )
            except Exception as e:
                print(f"⚠️  Failed to log baseline metrics to wandb: {e}")
        else:
            print("ℹ️  Baseline metrics (preview):", baseline_metrics)
    except Exception as e:
        print(f"⚠️  Baseline validation failed: {e}")
    finally:
        # Restore prior training mode for the model, but keep teacher in eval
        try:
            model.train(prev_mode)
            if getattr(model, "teacher_backbone", None) is not None:
                model.teacher_backbone.eval()
        except Exception:
            pass

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
