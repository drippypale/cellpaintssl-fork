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
import wandb

import source.augment as au
from source import SimCLR, get_jump_dataloaders


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
        domain_hidden: int = 128,
        freeze_encoder: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_domains = int(num_domains)
        self.adv_lambda = float(adv_lambda)
        self.freeze_encoder = bool(freeze_encoder)

        # Initialize GRL with the user-specified lambda
        self.grl = GradientReversal(lambd=self.adv_lambda)

        # Infer domain head input dimension from actual projector
        # Get the penultimate layer output dimension from the MLP
        in_dim = self.mlp[0].out_features  # First linear layer output dim

        self.domain_head = nn.Sequential(
            nn.Linear(in_dim, domain_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(domain_hidden, self.num_domains),
        )

        if self.freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        # Optimize only trainable parameters (projector + domain head; encoder frozen)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = self._get_lr_scheduler(optimizer)
        return [optimizer], [lr_scheduler]

    def _get_lr_scheduler(self, optimizer):
        from source import warmup_scheduler

        return warmup_scheduler.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            warmup_start_lr=1e-6,
            max_epochs=self.hparams.max_epochs,
            eta_min=self.hparams.lr_final_value,
        )

    def _forward_embeddings(self, imgs):
        # imgs: concatenated tensor of shape [2*B, C, H, W]
        backbone_feats = self.backbone(imgs)
        # Penultimate projector features: Linear -> ReLU
        penultimate = self.mlp[1](self.mlp[0](backbone_feats))
        # Final contrastive representation
        feats = self.mlp[2](penultimate)
        return feats, penultimate

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

    def _domain_loss_and_metrics(self, penultimate, domain_targets, mode: str):
        # Apply GRL before the domain head
        rev = self.grl(penultimate)
        logits = self.domain_head(rev)
        loss = F.cross_entropy(logits, domain_targets)

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
        """Update GRL lambda using Ganin schedule"""
        if hasattr(self, "trainer") and self.trainer is not None:
            current_step = (
                self.current_epoch * self.trainer.num_training_batches + batch_idx
            )
            total_steps = self.hparams.max_epochs * self.trainer.num_training_batches
            p = current_step / total_steps
            # Ganin schedule: ramps up from 0 to adv_lambda
            lam = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * self.adv_lambda
            self.grl.lambd = lam

            # Log lambda progress every 10 batches
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

        feats, penultimate = self._forward_embeddings(imgs)

        # SimCLR loss and metrics
        simclr_loss = self._simclr_loss_and_metrics(feats, mode="train")

        # Domain targets repeated for both views
        domain_targets = (
            domain_labels.detach().clone().to(device=self.device, dtype=torch.long)
        )
        domain_targets = torch.cat([domain_targets, domain_targets], dim=0)

        domain_loss = self._domain_loss_and_metrics(
            penultimate, domain_targets, mode="train"
        )

        # Combine losses without additional weighting (lambda is in GRL)
        total_loss = simclr_loss + domain_loss
        self.log("train_total_loss", total_loss, prog_bar=True)

        # Log GRL lambda for tracking
        self.log("train_grl_lambda", self.adv_lambda, prog_bar=False)

        # Log detailed progress every 10 batches
        if batch_idx % 10 == 0:
            print(
                f"    [LOSSES] SimCLR: {simclr_loss:.4f}, Domain: {domain_loss:.4f}, Total: {total_loss:.4f}"
            )
            print(f"    [GRL] Lambda: {self.adv_lambda:.3f}")

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Progress tracking for validation
        if batch_idx % 2 == 0:  # Print every 2 validation batches
            print(f"  [VAL] Epoch {self.current_epoch}, Batch {batch_idx}")

        imgs, _, domain_labels = batch
        imgs = torch.cat(imgs, dim=0)
        feats, penultimate = self._forward_embeddings(imgs)
        _ = self._simclr_loss_and_metrics(feats, mode="val")
        domain_targets = (
            torch.tensor(domain_labels, device=self.device, dtype=torch.long)
            .detach()
            .clone()
        )
        domain_targets = torch.cat([domain_targets, domain_targets], dim=0)
        _ = self._domain_loss_and_metrics(penultimate, domain_targets, mode="val")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        print(f"✅ Epoch {self.current_epoch} training completed")
        print(f"  - Final GRL lambda: {self.adv_lambda:.3f}")

        # Log epoch-level metrics to wandb
        if hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "log"
        ):
            # Log current learning rate
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log("train_learning_rate", current_lr, prog_bar=False)

            # Log epoch summary
            self.log("train_epoch", self.current_epoch, prog_bar=False)

        print()

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        print(f"✅ Epoch {self.current_epoch} validation completed")

        # Log validation epoch metrics to wandb
        if hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "log"
        ):
            self.log("val_epoch", self.current_epoch, prog_bar=False)

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
    # Adam parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    # Model parameters
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of contrastive projector output",
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
        "--domain_hidden", type=int, default=128, help="Hidden units in domain head"
    )
    parser.add_argument(
        "--no_freeze_encoder",
        action="store_true",
        help="Do not freeze the encoder backbone",
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
    print()

    # Seed and determinism
    print("🔧 SETTING UP ENVIRONMENT...")
    pl.seed_everything(args.seed)
    os.makedirs(ckpt_path, exist_ok=True)
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
            # Note: batch_sampler handles drop_last internally by only yielding complete batches
        )
        print("  - Balanced batch sampling enabled")

    num_domains = len(batch_to_index)
    print(f"  - Number of domains: {num_domains}")
    print(f"  - Domain mapping: {batch_to_index}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print()

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
                "weight_decay": weight_decay,
                "hidden_dim": hidden_dim,
                "temperature": temperature,
                "adv_lambda": adv_lambda,
                "domain_hidden": domain_hidden,
                "freeze_encoder": freeze_encoder,
                "balanced_batches": balanced_batches,
                "num_workers": num_workers,
                "train_ratio": train_ratio,
                "num_domains": num_domains,
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
        log_every_n_steps=5,  # Log every 10 steps instead of default 50
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                save_top_k=-1,
                monitor=None,
                every_n_epochs=every_n_epochs,
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
    model = SimCLRWithGRL(
        num_domains=num_domains,
        adv_lambda=adv_lambda,
        domain_hidden=domain_hidden,
        freeze_encoder=freeze_encoder,
        max_epochs=max_epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        temperature=temperature,
        weight_decay=weight_decay,
        vit=args.arch,
    )
    print(f"  - Architecture: {args.arch}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Temperature: {temperature}")
    print(f"  - Domain hidden: {domain_hidden}")
    print(f"  - Freeze encoder: {freeze_encoder}")
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

    trainer.fit(model, train_loader, val_loader)

    print("=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
