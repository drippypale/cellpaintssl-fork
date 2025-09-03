#!/usr/bin/env python3
"""
Inference script for GRL-SimCLR models trained on JUMP data.
Extracts embeddings from GRL-SimCLR checkpoints and saves them in the format expected by evaluate.py.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import warnings

# Import our custom modules
from source.jump_data import get_jump_dataloaders
from train_grl_jump import SimCLRWithGRL
from source import augment as au

warnings.filterwarnings("ignore")


class GRLSimCLREmbeddingNet(torch.nn.Module):
    """
    Wrapper to extract embeddings from GRL-SimCLR model.
    Only uses backbone + projector, ignores domain head and GRL.
    """

    def __init__(self, grl_model):
        super().__init__()
        self.backbone = grl_model.backbone
        # Adapter-related modules from GRL model (may not exist on old ckpts)
        self.adapter = getattr(grl_model, "adapter", None)
        self.adapter_scale = float(getattr(grl_model, "adapter_scale", 0.0))
        self.mlp = getattr(grl_model, "mlp", None)

    def forward(self, x):
        # Extract backbone features and apply residual adapter if available
        backbone_feats = self.backbone(x)
        if self.adapter is not None and self.adapter_scale != 0.0:
            adapted = backbone_feats + self.adapter_scale * self.adapter(backbone_feats)
            return adapted
        return backbone_feats


def load_grl_simclr_model(checkpoint_path, arch="vit_small_patch16_224"):
    """
    Load GRL-SimCLR model and create embedding extractor.
    """
    print(f"Loading GRL-SimCLR checkpoint from: {checkpoint_path}")

    # First, load the checkpoint to get the hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Prefer Lightning to restore the full model and its saved hyperparameters
    try:
        model = SimCLRWithGRL.load_from_checkpoint(checkpoint_path, strict=False)
        print("  - Restored model and hyperparameters from checkpoint")
    except Exception as e:
        print(
            f"⚠️  Direct restore failed ({e}). Falling back to manual init from hparams..."
        )
        # Extract minimal hyperparameters for a best-effort load
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            model = SimCLRWithGRL(
                num_domains=hparams.get("num_domains", 2),
                adv_lambda=hparams.get("adv_lambda", 1.0),
                domain_hidden=hparams.get("domain_hidden", 128),
                freeze_encoder=hparams.get("freeze_encoder", True),
                adapter_hidden=hparams.get("adapter_hidden", 384),
                adapter_scale=hparams.get("adapter_scale", 0.1),
                max_epochs=hparams.get("max_epochs", 1),
                warmup_epochs=hparams.get("warmup_epochs", 1),
                lr_final_value=hparams.get("lr_final_value", 1e-6),
                lr=hparams.get("lr", 1e-3),
                hidden_dim=hparams.get("hidden_dim", 128),
                temperature=hparams.get("temperature", 0.2),
                weight_decay=hparams.get("weight_decay", 0.1),
                vit=hparams.get("vit", arch),
                domain_loss_weight=hparams.get("domain_loss_weight", 1.0),
            )
            missing, unexpected = model.load_state_dict(
                checkpoint.get("state_dict", checkpoint), strict=False
            )
            print(
                f"  - Manual state load: missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            raise RuntimeError("Checkpoint does not contain required hyperparameters.")

    # Create embedding extractor (backbone + projector only)
    embedding_net = GRLSimCLREmbeddingNet(model)
    embedding_net.eval()

    print("✅ GRL-SimCLR model loaded successfully")
    print(f"  - Backbone: {model.backbone.__class__.__name__}")
    try:
        print(f"  - Projector output dim: {model.mlp[-1].out_features}")
    except Exception:
        pass

    return embedding_net


def extract_embeddings(
    model,
    dataloader,
    device,
    operation="mean",
    means=None,
    stds=None,
    crop_size: int = 224,
):
    """
    Extract embeddings from a dataloader and aggregate them.
    """
    model.to(device)
    model.eval()

    all_embeddings = []
    all_metadata = []

    print(f"Extracting embeddings with operation: {operation}")
    # Default normalization: match training in train_grl_jump.py
    do_norm = True
    if means is None and stds is None:
        means = [0.13849893, 0.18710597, 0.1586524, 0.15757588, 0.08674719]
        stds = [0.13005716, 0.15461144, 0.15929441, 0.16021383, 0.16686504]
    elif means == [] or stds == []:
        do_norm = False
    if do_norm:
        means_t = torch.tensor(means, device=device).view(1, -1, 1, 1)
        stds_t = torch.tensor(stds, device=device).view(1, -1, 1, 1)

    # Prepare cell-aware cropper (mandatory)
    cropper = au.RandomCropWithCells(size=crop_size)

    with torch.no_grad():
        for batch_idx, (views, metadata, _) in enumerate(
            tqdm(dataloader, desc="Extracting embeddings")
        ):
            # views is a list of two augmented views, we only need one for inference
            if isinstance(views, list):
                images = views[0]  # Take first view only
            else:
                images = views

            # Create one cell-aware crop per image (C, crop_size, crop_size)
            B = images.shape[0]
            crops = []
            for b in range(B):
                img_b = images[b].detach().cpu()
                crop_b = cropper(img_b, thresh=None)
                crops.append(crop_b)
            crops = torch.stack(crops, dim=0)  # [B, C, crop_size, crop_size]
            crops = crops.to(device)

            # Normalize per channel
            if do_norm:
                crops = (crops - means_t) / stds_t

            # Extract embeddings
            emb = model(crops)  # [B, D]

            # Convert to numpy
            embeddings_np = emb.cpu().numpy()

            # Store embeddings and metadata
            for i in range(len(embeddings_np)):
                all_embeddings.append(embeddings_np[i])
                all_metadata.append(
                    {
                        "source": metadata["source"][i],
                        "batch": metadata["batch"][i],
                        "plate": metadata["plate"][i],
                        "well": metadata["well"][i],
                        "site": metadata["site"][i],
                        "compound": metadata["compound"][i],
                        "smiles": metadata["smiles"][i],
                        "pert_type": metadata["pert_type"][i],
                    }
                )

    return all_embeddings, all_metadata


def aggregate_embeddings_to_well_level(embeddings, metadata, operation="mean"):
    """
    Aggregate embeddings from site level to well level.
    """
    print(f"Aggregating embeddings to well level using {operation}")

    # Create DataFrame for easier aggregation
    df = pd.DataFrame(metadata)
    df["embeddings"] = embeddings

    # Group by well and aggregate embeddings
    well_groups = df.groupby(["batch", "plate", "well"])

    well_embeddings = []
    well_metadata = []

    for (batch, plate, well), group in well_groups:
        # Aggregate embeddings across sites
        if operation == "mean":
            well_emb = np.mean(group["embeddings"].tolist(), axis=0)
        elif operation == "median":
            well_emb = np.median(group["embeddings"].tolist(), axis=0)
        else:
            raise ValueError(f"Unknown aggregation operation: {operation}")

        well_embeddings.append(well_emb)

        # Take first metadata entry for the well
        well_metadata.append(
            {
                "batch": batch,
                "plate": plate,
                "well": well,
                "source": group["source"].iloc[0],
                "compound": group["compound"].iloc[0],
                "smiles": group["smiles"].iloc[0],
                "pert_type": group["pert_type"].iloc[0],
            }
        )

    return well_embeddings, well_metadata


def create_well_features_csv(well_embeddings, well_metadata, output_path):
    """
    Create well_features.csv in the format expected by evaluate.py
    """
    print(f"Creating well_features.csv at: {output_path}")

    # Convert embeddings to DataFrame
    embedding_dim = len(well_embeddings[0])
    embedding_cols = [f"emb{i}" for i in range(embedding_dim)]

    embeddings_df = pd.DataFrame(well_embeddings, columns=embedding_cols)
    metadata_df = pd.DataFrame(well_metadata)

    # Combine metadata and embeddings
    well_features_df = pd.concat([metadata_df, embeddings_df], axis=1)

    # Add required columns for evaluation
    # Note: These might need to be filled based on your actual data
    well_features_df["perturbation_id"] = well_features_df["compound"].fillna("DMSO")
    well_features_df["target"] = well_features_df["compound"].fillna(
        "DMSO"
    )  # You might need to map this properly

    # Reorder columns to match expected format
    required_cols = [
        "batch",
        "plate",
        "well",
        "perturbation_id",
        "target",
    ] + embedding_cols
    available_cols = [col for col in required_cols if col in well_features_df.columns]

    well_features_df = well_features_df[available_cols]

    # Save to CSV
    well_features_df.to_csv(output_path, index=False)
    print(
        f"✅ Saved well_features.csv with {len(well_features_df)} wells and {embedding_dim} features"
    )

    return well_features_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from GRL-SimCLR model"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to GRL-SimCLR checkpoint"
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
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for embeddings"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--operation",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation operation for embeddings",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (for debugging)",
    )
    parser.add_argument(
        "--arch", type=str, default="vit_small_patch16_224", help="Model architecture"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable input normalization (defaults to enabled)",
    )
    parser.add_argument(
        "--size", type=int, default=224, help="Crop size for RandomCropWithCells"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 GRL-SIMCLR INFERENCE")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Submission CSV: {args.submission_csv}")
    print(f"Images base path: {args.images_base_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Operation: {args.operation}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_grl_simclr_model(args.ckpt, args.arch)

    # Create dataloaders (no transforms needed for inference)
    print("Creating dataloaders...")
    train_loader, val_loader, batch_to_index = get_jump_dataloaders(
        submission_csv=args.submission_csv,
        images_base_path=args.images_base_path,
        transform=None,  # No transforms for inference
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=1.0,  # Use all data for inference
        max_samples=args.max_samples,
        with_domain_labels=True,
    )

    print(f"Dataset loaded: {len(train_loader.dataset)} samples")
    print(f"Number of domains: {len(batch_to_index)}")

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, metadata = extract_embeddings(
        model,
        train_loader,
        device,
        args.operation,
        None if not args.no_normalize else [],
        None if not args.no_normalize else [],
        crop_size=args.size,
    )

    # Aggregate to well level
    print("Aggregating to well level...")
    well_embeddings, well_metadata = aggregate_embeddings_to_well_level(
        embeddings, metadata, args.operation
    )

    # Create well_features.csv
    output_path = os.path.join(args.output_dir, "well_features.csv")
    well_features_df = create_well_features_csv(
        well_embeddings, well_metadata, output_path
    )

    print("=" * 80)
    print("✅ INFERENCE COMPLETED!")
    print("=" * 80)
    print(f"Output saved to: {output_path}")
    print(f"Total wells: {len(well_features_df)}")
    print(f"Embedding dimension: {len(well_embeddings[0])}")
    print()
    print("Next steps:")
    print("1. Copy this output to SSL_data/embeddings/grl_comparison/GRL-SimCLR/")
    print("2. Copy repository models to the same directory")
    print("3. Run: python evaluate.py -i grl_comparison -b SSL_data/embeddings")


if __name__ == "__main__":
    main()
