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
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Import our custom modules
from source.jump_data import get_jump_dataloaders
from train_grl_jump import SimCLRWithGRL
from source import utils


class GRLSimCLREmbeddingNet(torch.nn.Module):
    """
    Wrapper to extract embeddings from GRL-SimCLR model.
    Only uses backbone + projector, ignores domain head and GRL.
    """

    def __init__(self, grl_model):
        super().__init__()
        self.backbone = grl_model.backbone
        self.mlp = grl_model.mlp

    def forward(self, x):
        # Extract backbone features
        backbone_feats = self.backbone(x)
        # Project to final embeddings (same as SimCLR)
        embeddings = self.mlp(backbone_feats)
        return embeddings


def load_grl_simclr_model(checkpoint_path, arch="vit_small_patch16_224"):
    """
    Load GRL-SimCLR model and create embedding extractor.
    """
    print(f"Loading GRL-SimCLR checkpoint from: {checkpoint_path}")

    # First, load the checkpoint to get the hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract hyperparameters from checkpoint
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
        num_domains = hparams.get("num_domains", 10)  # Default to 10 if not found
        adv_lambda = hparams.get("adv_lambda", 1.0)
        domain_hidden = hparams.get("domain_hidden", 128)
        freeze_encoder = hparams.get("freeze_encoder", True)
        max_epochs = hparams.get("max_epochs", 1)
        lr = hparams.get("lr", 1e-3)
        hidden_dim = hparams.get("hidden_dim", 128)
        temperature = hparams.get("temperature", 0.2)
        weight_decay = hparams.get("weight_decay", 0.1)
    else:
        # Fallback values if hyperparameters not found
        print("⚠️  Hyperparameters not found in checkpoint, using defaults")
        num_domains = 10  # Default to 10 domains for JUMP
        adv_lambda = 1.0
        domain_hidden = 128
        freeze_encoder = True
        max_epochs = 1
        lr = 1e-3
        hidden_dim = 128
        temperature = 0.2
        weight_decay = 0.1

    print(f"  - Extracted hyperparameters:")
    print(f"    - num_domains: {num_domains}")
    print(f"    - adv_lambda: {adv_lambda}")
    print(f"    - domain_hidden: {domain_hidden}")
    print(f"    - hidden_dim: {hidden_dim}")

    # Load the full GRL-SimCLR model with correct hyperparameters
    model = SimCLRWithGRL.load_from_checkpoint(
        checkpoint_path,
        num_domains=num_domains,
        adv_lambda=adv_lambda,
        domain_hidden=domain_hidden,
        freeze_encoder=freeze_encoder,
        max_epochs=max_epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        temperature=temperature,
        weight_decay=weight_decay,
        vit=arch,
    )

    # Create embedding extractor (backbone + projector only)
    embedding_net = GRLSimCLREmbeddingNet(model)
    embedding_net.eval()

    print("✅ GRL-SimCLR model loaded successfully")
    print(f"  - Backbone: {model.backbone.__class__.__name__}")
    print(f"  - Projector output dim: {model.mlp[-1].out_features}")

    return embedding_net


def extract_embeddings(model, dataloader, device, operation="mean"):
    """
    Extract embeddings from a dataloader and aggregate them.
    """
    model.to(device)
    model.eval()

    all_embeddings = []
    all_metadata = []

    print(f"Extracting embeddings with operation: {operation}")

    with torch.no_grad():
        for batch_idx, (views, metadata, _) in enumerate(
            tqdm(dataloader, desc="Extracting embeddings")
        ):
            # views is a list of two augmented views, we only need one for inference
            if isinstance(views, list):
                images = views[0]  # Take first view only
            else:
                images = views

            images = images.to(device)

            # Extract embeddings
            embeddings = model(images)  # Shape: [batch_size, embedding_dim]

            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()

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
        model, train_loader, device, args.operation
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
