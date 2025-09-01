#!/usr/bin/env python3
"""
GRL-SimCLR inference script that follows the original inference.py pattern
to work with the standard evaluation pipeline and downloaded JUMP data.
"""

import os
import argparse
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from source.models import EmbeddingNet
from source.utils import log_and_print
from source.inference_utils import (
    forward_inference,
    aggregate_embeddings_plate,
    post_proc,
)
from source import MergedChannelsDataset

# Import our GRL-SimCLR model
from train_grl_jump import SimCLRWithGRL, GradientReversal


class GRLSimCLREmbeddingNet(EmbeddingNet):
    """Embedding network for GRL-SimCLR that extracts features from the projector (not domain head)"""

    def __init__(self, grl_model):
        super().__init__(grl_model)
        # Override to use the projector output instead of backbone
        self.backbone = grl_model.backbone
        self.mlp = grl_model.mlp  # This is the SimCLR projector

    def forward(self, x):
        """Extract embeddings from the SimCLR projector (domain-invariant features)"""
        backbone_feats = self.backbone(x)
        embeddings = self.mlp(backbone_feats)  # Projector output
        return embeddings


def load_grl_simclr_model(checkpoint_path, arch="vit_small_patch16_224"):
    """
    Load GRL-SimCLR model from checkpoint.
    """
    print(f"Loading GRL-SimCLR model from: {checkpoint_path}")

    # Load checkpoint to extract hyperparameters
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract hyperparameters
    if isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
        hparams = ckpt["hyper_parameters"]
        print(f"Found hyperparameters: {hparams}")

        # Extract model parameters
        max_epochs = hparams.get("max_epochs", 1)
        lr = hparams.get("lr", 1e-3)
        hidden_dim = hparams.get(
            "hidden_dim", 384
        )  # Default to 384 to match original SimCLR
        temperature = hparams.get("temperature", 0.2)
        weight_decay = hparams.get("weight_decay", 0.1)
        num_domains = hparams.get("num_domains", 5)
        adv_lambda = hparams.get("adv_lambda", 1.0)
        domain_hidden = hparams.get("domain_hidden", 128)
        freeze_encoder = hparams.get("freeze_encoder", True)
        vit = hparams.get("vit", arch)
    else:
        print("⚠️  No hyperparameters found, using defaults")
        max_epochs = 1
        lr = 1e-3
        hidden_dim = 384
        temperature = 0.2
        weight_decay = 0.1
        num_domains = 5
        adv_lambda = 1.0
        domain_hidden = 128
        freeze_encoder = True
        vit = arch

    # Create GRL-SimCLR model
    grl_model = SimCLRWithGRL(
        num_domains=num_domains,
        adv_lambda=adv_lambda,
        domain_hidden=domain_hidden,
        freeze_encoder=freeze_encoder,
        max_epochs=max_epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        temperature=temperature,
        weight_decay=weight_decay,
        vit=vit,
    )

    # Load checkpoint weights
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = grl_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")

    # Create embedding network that extracts from projector
    embednet = GRLSimCLREmbeddingNet(grl_model)

    print(f"✅ GRL-SimCLR model loaded successfully")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Architecture: {vit}")
    print(f"  - Embedding dim: {hidden_dim}")

    return embednet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", help="GRL-SimCLR checkpoint file", type=str, required=True
    )
    parser.add_argument(
        "--valset",
        help="Validation csv file path",
        default="data/JUMP_valset.csv",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--operation",
        help="How to aggregate crops, FOV, and perturbations",
        default="mean",
        type=str,
    )
    parser.add_argument(
        "--norm_method",
        help="How to norm features",
        default="all",
        choices=[
            "standardize",
            "mad_robustize",
            "spherize",
            "spherize_mad_robustize",
            "mad_robustize_spherize",
            "spherize_standardize",
            "standardize_spherize",
            "no_post_proc",
            "all",
        ],
        type=str,
    )
    parser.add_argument("-o", "--outdir", help="Output directory", required=True)
    parser.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--size", nargs="?", const=224, type=int, default=224)
    parser.add_argument("--stride", nargs="?", const=None, type=int, default=None)
    parser.add_argument("--l2norm", default=False, action="store_true")
    parser.add_argument(
        "--arch", type=str, default="vit_small_patch16_224", help="Model architecture"
    )

    args = parser.parse_args()

    print(f"🚀 GRL-SimCLR Inference")
    print(f"  - Checkpoint: {args.ckpt}")
    print(f"  - Validation set: {args.valset}")
    print(f"  - Output dir: {args.outdir}")
    print(f"  - Norm method: {args.norm_method}")
    print(f"  - Operation: {args.operation}")
    print()

    # Handle norm method
    if args.norm_method == "all":
        norm_method = [
            "standardize",
            "mad_robustize",
            "spherize",
            "spherize_mad_robustize",
            "mad_robustize_spherize",
            "spherize_standardize",
            "standardize_spherize",
            "no_post_proc",
        ]
    else:
        norm_method = [args.norm_method]

    # Crop size and stride
    crop_size = args.size
    stride = args.stride if args.stride is not None else crop_size

    # Output directories
    outdirs = []
    if args.norm_method == "all":
        for m in norm_method:
            outdir = os.path.join(args.outdir, m)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outdirs.append(outdir)
    else:
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdirs.append(outdir)

    os.system(f"chmod -R 777 {args.outdir}")

    # Load validation set
    print(f"📊 Loading validation set: {args.valset}")
    val_df = pd.read_csv(args.valset)
    all_plates = list(val_df["plate"].drop_duplicates())
    print(f"  - Found {len(all_plates)} plates")
    print(f"  - Total samples: {len(val_df)}")

    # Columns to keep
    cols_to_keep = ["batch", "plate", "well", "perturbation_id", "target"]

    # Channel stats (same as original)
    means = torch.tensor([0.12528631, 0.17596765, 0.14736995, 0.13445823, 0.08349566])
    sds = torch.tensor([0.12594905, 0.15605405, 0.16031352, 0.15751939, 0.15773378])

    # Load GRL-SimCLR model
    print(f"🤖 Loading GRL-SimCLR model...")
    embednet = load_grl_simclr_model(args.ckpt, arch=args.arch)

    # Setup device and model
    device = torch.device(
        "cuda:" + str(args.gpus[0]) if torch.cuda.is_available() else "cpu"
    )
    model = (
        torch.nn.DataParallel(embednet, device_ids=args.gpus)
        if torch.cuda.is_available()
        else embednet
    )
    model.to(device)
    model.eval()

    print(f"  - Device: {device}")
    print(f"  - Model loaded successfully")
    print()

    # Run inference for every plate
    print(f"🔍 Running inference for {len(all_plates)} plates...")
    transf = transforms.Compose([transforms.Normalize(means, sds)])
    all_embs = []

    for plt in all_plates:
        print(f"  📋 Processing plate: {plt}")
        val_df_sub = val_df[val_df["plate"] == plt].copy()
        val_df_sub = val_df_sub.reset_index(drop=True)

        platedata = MergedChannelsDataset(val_df_sub, transform=transf, inference=True)
        plateloader = DataLoader(
            platedata,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
        )

        plate_embs = []
        for crops_with_metadata in tqdm(plateloader, desc=f"Plate {plt}"):
            with torch.no_grad():
                crop_embs = forward_inference(
                    model,
                    crops_with_metadata["crops"],
                    crops_with_metadata["labels"],
                    device,
                )
                plate_embs.append(crop_embs.cpu().numpy())

        embeddings = aggregate_embeddings_plate(
            plate_dfr=val_df_sub,
            plate_embs=plate_embs,
            my_cols=cols_to_keep,
            operation=args.operation,
        )
        all_embs.append(embeddings)
        log_and_print(f"Inference for plate {plt} finished successfully.")

    # Concatenate all plate embeddings
    embedding_df = pd.concat(all_embs, ignore_index=True)
    print(f"✅ All plates processed. Total embeddings: {len(embedding_df)}")

    # Postprocessing
    print(f"🔄 Postprocessing embeddings...")
    print(f"  - Methods: {norm_method}")
    print(f"  - Aggregation: {args.operation}")

    for outdir, norm in zip(outdirs, norm_method):
        print(f"  📁 Processing {norm} -> {outdir}")

        # Handle combined normalization methods
        if "spherize_" in norm:
            well_embs, _ = post_proc(
                embedding_df,
                val_df,
                operation=args.operation,
                norm_method="spherize",
                l2_norm=args.l2norm,
            )
            embeddings_proc_well, embeddings_proc_agg = post_proc(
                well_embs,
                val_df,
                operation=args.operation,
                norm_method=norm.replace("spherize_", ""),
            )
        elif "_spherize" in norm:
            well_embs, _ = post_proc(
                embedding_df,
                val_df,
                operation=args.operation,
                norm_method=norm.replace("_spherize", ""),
                l2_norm=args.l2norm,
            )
            embeddings_proc_well, embeddings_proc_agg = post_proc(
                well_embs, val_df, operation=args.operation, norm_method="spherize"
            )
        else:
            # Single normalization method
            embeddings_proc_well, embeddings_proc_agg = post_proc(
                embedding_df,
                val_df,
                operation=args.operation,
                norm_method=norm,
                l2_norm=args.l2norm,
            )

        # Save results
        csv_path_well = os.path.join(outdir, "well_features.csv")
        csv_path_agg = os.path.join(outdir, "agg_features.csv")
        embeddings_proc_well.to_csv(csv_path_well, index=False)
        embeddings_proc_agg.to_csv(csv_path_agg, index=False)

        print(f"    💾 Saved: {csv_path_well}")
        print(f"    💾 Saved: {csv_path_agg}")

    print("🎉 GRL-SimCLR inference completed successfully!")
    print(f"📁 Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
