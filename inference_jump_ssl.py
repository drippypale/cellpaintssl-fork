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
from source.inference_utils import post_proc
from source.mae.load_mae import load_pretrained_mae
from source.dino.utils import load_pretrained_dino
import source.dino.vision_transformer as vits
import warnings

warnings.filterwarnings("ignore")


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
    parser.add_argument(
        "--norm_method",
        type=str,
        default="spherize",
        choices=[
            "standardize",
            "mad_robustize",
            "spherize",
            "spherize_mad_robustize",
            "mad_robustize_spherize",
            "spherize_standardize",
            "standardize_spherize",
            "no_post_proc",
        ],
    )
    parser.add_argument("--l2norm", action="store_true")
    # XGBoost-based batch importance & masking
    parser.add_argument(
        "--xgb_batch_mask_fraction",
        type=float,
        default=0.0,
        help="If >0, train XGBoost to predict batch on well embeddings, drop top fraction of most important features.",
    )
    parser.add_argument(
        "--xgb_top_k_plot",
        type=int,
        default=40,
        help="Top-K features to plot in importance barplot.",
    )
    parser.add_argument(
        "--xgb_random_state",
        type=int,
        default=42,
        help="Random state for XGBoost training.",
    )
    # Batch-leakage feature filtering
    parser.add_argument(
        "--drop_batch_fraction",
        type=float,
        default=0.0,
        help="Fraction of most batch-predictive embedding features to drop (0-1).",
    )
    parser.add_argument(
        "--drop_batch_cv",
        type=int,
        default=3,
        help="K folds for cross-validated logistic regression per feature.",
    )
    parser.add_argument(
        "--drop_batch_max_features",
        type=int,
        default=None,
        help="Optional hard cap on number of features to drop (after fraction).",
    )

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

    # Save raw well features
    out_csv = os.path.join(args.output_dir, "well_features.csv")
    print(f"Saving raw to {out_csv}")
    df = save_well_features(well_embs, well_meta, out_csv)
    print(f"✅ Saved {len(df)} wells with embedding dim {len(well_embs[0])}")

    # Optional: identify and drop batch-predictive features via per-feature logistic regression
    if args.drop_batch_fraction and args.drop_batch_fraction > 0.0:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            import numpy as np
            import pandas as pd

            emb_cols = [c for c in df.columns if c.startswith("emb")]
            if len(emb_cols) == 0:
                print("No embedding columns found for feature dropping; skipping.")
            else:
                print(
                    f"Scoring {len(emb_cols)} features by batch predictability (CV={args.drop_batch_cv})"
                )
                X = df[emb_cols].to_numpy()
                y = df["batch"].astype(str).values
                # Use liblinear for small dimensional single-feature fits; multinomial auto-handled
                lr = LogisticRegression(max_iter=200, solver="liblinear")
                cv = StratifiedKFold(
                    n_splits=max(2, args.drop_batch_cv), shuffle=True, random_state=42
                )
                scores = []
                for j, col in enumerate(emb_cols):
                    xj = X[:, j].reshape(-1, 1)
                    try:
                        acc = cross_val_score(
                            lr, xj, y, cv=cv, scoring="accuracy"
                        ).mean()
                    except Exception:
                        acc = 0.0
                    scores.append(acc)
                scores = np.array(scores)
                # Rank descending by accuracy
                order = np.argsort(-scores)
                n_drop_by_frac = int(
                    np.floor(len(emb_cols) * float(args.drop_batch_fraction))
                )
                n_drop = n_drop_by_frac
                if args.drop_batch_max_features is not None:
                    n_drop = min(n_drop, int(args.drop_batch_max_features))
                n_drop = max(0, min(n_drop, len(emb_cols)))
                drop_idx = order[:n_drop]
                drop_cols = [emb_cols[i] for i in drop_idx]
                keep_cols = [c for c in emb_cols if c not in drop_cols]
                print(
                    f"Dropping {len(drop_cols)}/{len(emb_cols)} features (~{100.0 * len(drop_cols) / max(1, len(emb_cols)):.1f}%)"
                )
                # Save a report
                rep = pd.DataFrame(
                    {
                        "feature": emb_cols,
                        "cv_accuracy": scores,
                        "rank": (-scores).argsort().argsort() + 1,
                        "dropped": [c in drop_cols for c in emb_cols],
                    }
                ).sort_values("cv_accuracy", ascending=False)
                rep_path = os.path.join(args.output_dir, "batch_feature_scores.csv")
                rep.to_csv(rep_path, index=False)
                print(f"Saved batch feature scores to {rep_path}")
                # Apply mask to raw df for downstream postprocessing
                df = pd.concat(
                    [
                        df[[c for c in df.columns if not c.startswith("emb")]],
                        df[keep_cols],
                    ],
                    axis=1,
                )
        except Exception as e:
            print(f"Warning: batch-predictive feature dropping failed: {e}")

    # Post-process and save normalized well and aggregate features
    print(f"Postprocessing embeddings method: {args.norm_method}")

    # Optional: XGBoost-based batch importance scoring and masking
    if args.xgb_batch_mask_fraction and args.xgb_batch_mask_fraction > 0.0:
        try:
            try:
                import xgboost as xgb
            except Exception as e:
                raise RuntimeError(
                    f"xgboost not available. Please install it to use --xgb_batch_mask_fraction. Error: {e}"
                )

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.preprocessing import LabelEncoder

            emb_cols = [c for c in df.columns if c.startswith("emb")]
            if len(emb_cols) == 0:
                print("No embedding columns found for XGBoost masking; skipping.")
            else:
                X = df[emb_cols].to_numpy()
                y_raw = df["batch"].astype(str).fillna("unknown").values
                # Guard: need at least two batches to train a classifier
                if np.unique(y_raw).size < 2:
                    print(
                        "XGBoost masking skipped: only one unique batch label present."
                    )
                    raise SystemExit(0)
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
                    random_state=int(args.xgb_random_state),
                    n_jobs=4,
                )
                clf.fit(X, y)

                # Get importance (gain preferred; fallback to weight)
                booster = clf.get_booster()
                try:
                    score_dict = booster.get_score(importance_type="gain")
                except Exception:
                    score_dict = booster.get_score(importance_type="weight")

                # Map xgb feature names (f0,f1,...) back to emb_cols
                importances = np.zeros(len(emb_cols), dtype=float)
                for k, v in score_dict.items():
                    try:
                        idx = int(k[1:])  # 'f123' -> 123
                        if 0 <= idx < len(importances):
                            importances[idx] = float(v)
                    except Exception:
                        continue

                # Save importance CSV
                imp_df = pd.DataFrame(
                    {
                        "feature": emb_cols,
                        "importance": importances,
                    }
                ).sort_values("importance", ascending=False)
                imp_csv = os.path.join(
                    args.output_dir, "xgb_batch_feature_importance.csv"
                )
                imp_df.to_csv(imp_csv, index=False)
                print(f"Saved XGBoost feature importances to {imp_csv}")

                # Plot top-k importances
                top_k = int(max(1, args.xgb_top_k_plot))
                imp_top = imp_df.head(top_k)
                plt.figure(figsize=(8, max(4, int(0.25 * top_k))))
                plt.barh(imp_top["feature"][::-1], imp_top["importance"][::-1])
                plt.xlabel("XGBoost importance (gain)")
                plt.ylabel("Feature")
                plt.title("Top XGBoost feature importances for batch prediction")
                plt.tight_layout()
                plot_path = os.path.join(
                    args.output_dir, "xgb_batch_feature_importance_topk.png"
                )
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"Saved importance plot to {plot_path}")

                # Mask top fraction of features
                frac = float(args.xgb_batch_mask_fraction)
                n_drop = int(np.floor(len(emb_cols) * frac))
                n_drop = max(0, min(n_drop, len(emb_cols)))
                drop_cols = imp_df.head(n_drop)["feature"].tolist()
                keep_cols = [c for c in emb_cols if c not in drop_cols]
                print(
                    f"XGB-masking: dropping {len(drop_cols)}/{len(emb_cols)} features (~{100.0 * len(drop_cols) / max(1, len(emb_cols)):.1f}%)"
                )
                df = pd.concat(
                    [
                        df[[c for c in df.columns if not c.startswith("emb")]],
                        df[keep_cols],
                    ],
                    axis=1,
                )
        except Exception as e:
            print(f"Warning: XGBoost-based masking failed: {e}")
    embeddings_proc_well, embeddings_proc_agg = post_proc(
        df,
        df,
        operation=args.operation,
        norm_method=args.norm_method,
        l2_norm=args.l2norm,
    )
    norm_outdir = (
        os.path.join(args.output_dir, args.norm_method)
        if args.norm_method != "no_post_proc"
        else args.output_dir
    )
    os.makedirs(norm_outdir, exist_ok=True)
    csv_path_well = os.path.join(norm_outdir, "well_features.csv")
    csv_path_agg = os.path.join(norm_outdir, "agg_features.csv")
    embeddings_proc_well.to_csv(csv_path_well, index=False)
    embeddings_proc_agg.to_csv(csv_path_agg, index=False)


if __name__ == "__main__":
    main()
