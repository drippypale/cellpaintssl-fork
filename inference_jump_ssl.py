#!/usr/bin/env python3
"""
JUMP-compatible inference for baseline SSL methods (SimCLR, DINO, MAE).

- Loads models in the same way as the repository's inference.py
- Uses JUMP dataloader from source.jump_data to read images/metadata
- Extracts site-level embeddings and aggregates to well-level
- Saves well_features.csv compatible with evaluate.py
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from source import SimCLR
from source.models import EmbeddingNet
from source.jump_data import get_jump_dataloaders

# from source.inference_utils import post_proc  # kept for parity if extended
from source.mae.load_mae import load_pretrained_mae
from source.dino.utils import load_pretrained_dino
import source.dino.vision_transformer as vits


def build_embedder(model_name: str, arch: str, ckpt: str):
    model_name = model_name.lower()
    if model_name == "simclr":
        vit_arch = (
            "vit_small_patch16_224" if "small" in arch else "vit_base_patch16_224"
        )
        try:
            ssl_model = SimCLR.load_from_checkpoint(
                ckpt, map_location="cpu", vit=vit_arch
            )
        except Exception as e:
            # Fallback: manual CPU load of state_dict
            print(
                f"Warning: Lightning load failed ({e}). Falling back to manual load on CPU."
            )
            ckpt_obj = torch.load(ckpt, map_location="cpu")
            state_dict = ckpt_obj.get("state_dict", ckpt_obj)
            ssl_model = SimCLR(vit=vit_arch)
            missing, unexpected = ssl_model.load_state_dict(state_dict, strict=False)
            print(
                f"  - Manual load: missing={len(missing)} unexpected={len(unexpected)}"
            )
        return EmbeddingNet(ssl_model)
    elif model_name == "mae":
        return load_pretrained_mae(arch, ckpt)
    elif model_name == "dino":
        patch_size = 16
        checkpoint_key = "teacher"
        # arch passed like vit_small or vit_base; keep parity with repository
        model_arch = "vit_small" if "small" in arch else "vit_base"
        model_arch = (
            model_arch.split("_")[0] + "_x" + model_arch.split("_")[1]
            if "x" in arch
            else model_arch
        )
        model = vits.__dict__[model_arch](patch_size=patch_size, drop_path_rate=0.1)
        return load_pretrained_dino(
            model,
            ckpt,
            checkpoint_key,
            model_name=None,
            patch_size=patch_size,
        )
    else:
        raise ValueError("--model must be one of: simclr, dino, mae")


def extract_embeddings(embedder, dataloader, device, resize_to=224, normalize=True):
    embedder.to(device)
    embedder.eval()

    means = [0.13849893, 0.18710597, 0.1586524, 0.15757588, 0.08674719]
    stds = [0.13005716, 0.15461144, 0.15929441, 0.16021383, 0.16686504]
    means_t = torch.tensor(means, device=device).view(1, -1, 1, 1)
    stds_t = torch.tensor(stds, device=device).view(1, -1, 1, 1)

    all_embeddings = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Batch may be (views, metadata, domain_labels) or (views, metadata)
            if len(batch) == 3:
                views, metadata, _ = batch
            else:
                views, metadata = batch

            # Use first view
            images = views[0] if isinstance(views, list) else views
            images = images.to(device)

            # Resize and normalize
            if resize_to is not None:
                images = F.interpolate(
                    images,
                    size=(resize_to, resize_to),
                    mode="bilinear",
                    align_corners=False,
                )
            if normalize:
                images = (images - means_t) / stds_t

            embs = embedder(images)
            embs_np = embs.detach().cpu().numpy()

            for i in range(embs_np.shape[0]):
                all_embeddings.append(embs_np[i])
                all_metadata.append(
                    {
                        "source": metadata["source"][i],
                        "batch": metadata["batch"][i],
                        "plate": metadata["plate"][i],
                        "well": metadata["well"][i],
                        "site": metadata["site"][i],
                        "compound": metadata.get("compound", [""])[i]
                        if isinstance(metadata.get("compound", None), list)
                        else metadata.get("compound", ""),
                        "target": metadata.get("target", [""])[i]
                        if isinstance(metadata.get("target", None), list)
                        else metadata.get("target", ""),
                        "smiles": metadata.get("smiles", [""])[i]
                        if isinstance(metadata.get("smiles", None), list)
                        else metadata.get("smiles", ""),
                        "pert_type": metadata.get("pert_type", [""])[i]
                        if isinstance(metadata.get("pert_type", None), list)
                        else metadata.get("pert_type", ""),
                    }
                )

    return all_embeddings, all_metadata


def aggregate_to_well(embeddings, metadata, operation="mean"):
    df = pd.DataFrame(metadata)
    df["embeddings"] = embeddings

    well_embeddings = []
    well_metadata = []

    for (batch, plate, well), group in df.groupby(["batch", "plate", "well"]):
        arrs = group["embeddings"].tolist()
        if operation == "mean":
            agg = np.mean(arrs, axis=0)
        elif operation == "median":
            agg = np.median(arrs, axis=0)
        else:
            raise ValueError("operation must be one of: mean, median")
        well_embeddings.append(agg)
        well_metadata.append(
            {
                "batch": batch,
                "plate": plate,
                "well": well,
                "source": group["source"].iloc[0],
                "compound": group.get("compound", pd.Series([""])).iloc[0]
                if "compound" in group
                else "",
                "target": group.get("target", pd.Series([""])).iloc[0]
                if "target" in group
                else "",
                "smiles": group.get("smiles", pd.Series([""])).iloc[0]
                if "smiles" in group
                else "",
                "pert_type": group.get("pert_type", pd.Series([""])).iloc[0]
                if "pert_type" in group
                else "",
            }
        )

    return well_embeddings, well_metadata


def save_well_features(well_embeddings, well_metadata, out_csv):
    emb_dim = len(well_embeddings[0])
    emb_cols = [f"emb{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(well_embeddings, columns=emb_cols)
    meta_df = pd.DataFrame(well_metadata)
    df = pd.concat([meta_df, emb_df], axis=1)

    # Minimal columns for evaluate.py
    df["perturbation_id"] = df.get(
        "compound", pd.Series(["DMSO"]).repeat(len(df))
    ).fillna("DMSO")
    # Use target if present; otherwise fallback to compound
    if "target" not in df.columns:
        df["target"] = df.get("compound", pd.Series(["DMSO"]).repeat(len(df))).fillna(
            "DMSO"
        )

    keep_cols = ["batch", "plate", "well", "perturbation_id", "target"] + emb_cols
    df = df[keep_cols]
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="JUMP-compatible inference for SSL models"
    )
    parser.add_argument("--model", choices=["simclr", "dino", "mae"], required=True)
    parser.add_argument("--arch", default="vit_small_patch16_224", type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--submission_csv", required=True, type=str)
    parser.add_argument(
        "--images_base_path",
        type=str,
        default="/content/drive/MyDrive/jump_data/images",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--operation", choices=["mean", "median"], default="mean")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    embedder = build_embedder(args.model, args.arch, args.ckpt)

    print("Creating JUMP dataloader...")
    train_loader, _, _ = get_jump_dataloaders(
        submission_csv=args.submission_csv,
        images_base_path=args.images_base_path,
        transform=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=1.0,
        max_samples=args.max_samples,
        with_domain_labels=False,
    )

    print("Extracting embeddings...")
    embs, meta = extract_embeddings(
        embedder,
        train_loader,
        device,
        resize_to=args.size,
        normalize=not args.no_normalize,
    )

    print("Aggregating to well level...")
    well_embs, well_meta = aggregate_to_well(embs, meta, args.operation)

    out_csv = os.path.join(args.output_dir, "well_features.csv")
    print(f"Saving to {out_csv}")
    df = save_well_features(well_embs, well_meta, out_csv)
    print(f"✅ Saved {len(df)} wells with embedding dim {len(well_embs[0])}")


if __name__ == "__main__":
    main()
