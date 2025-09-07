"""
Take a list of csv embeddings and output evaluation metrics and plots
for aggregated and well-level features.

Evaluation metrics for batch-aggregated and consensus profiles
"""

import os
import argparse
import pandas as pd
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from source import utils
import source.eval as evl


def load_embedding_data(indir, embed, orf):
    embed_dir = os.path.join(indir, embed)
    csv_path = os.path.join(embed_dir, "well_features.csv")

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return pd.DataFrame(), "standard", []

    df = pd.read_csv(csv_path)

    # Remove DMSO wells
    df = df.query('perturbation_id != "DMSO"')

    if orf:
        df = df.query('control_type != "negcon"')

    if len(df) == 0:
        print(f"❌ No wells remaining after filtering for {embed}")
        return df, "standard", []

    feature_type = (
        "cellprofiler"
        if embed in ["CellProfiler", "Random", "Shuffle CP"]
        else "standard"
    )

    features, _ = utils.get_feature_cols(df, feature_type=feature_type)
    return df, feature_type, features


def compute_perturbation_accuracy(df, features):
    X = StandardScaler().fit_transform(df[features].to_numpy())
    y = LabelEncoder().fit_transform(df["perturbation_id"].values)
    # Simple perturbation accuracy
    ypred = evl.nearest_neighbor_classifier_NSBW(X, y, mode="NN", metric="cosine")
    acc = accuracy_score(y, ypred)
    # NSB perturbation accuracy
    batches = df["batch"].values
    ypred = evl.nearest_neighbor_classifier_NSBW(
        X, y, mode="NSB", batches=batches, metric="cosine"
    )
    NSB_acc = accuracy_score(y, ypred)
    return acc, NSB_acc


def compute_target_accuracy(df, features, label="target"):
    X = StandardScaler().fit_transform(df[features].to_numpy())
    y = LabelEncoder().fit_transform(df[label].values)
    # not-same-perturbation (NSP) target accuracy
    ypred = evl.nearest_neighbor_classifier_NSBW(
        X, y, mode="NSW", metric="cosine", wells=df["well"].values
    )
    NSP_acc = accuracy_score(y, ypred)
    # not-same-batch-or-perturbation (NSBP) target accuracy
    batches = df["batch"].values
    ypred = evl.nearest_neighbor_classifier_NSBW(
        X, y, mode="NSBW", batches=batches, metric="cosine", wells=df["well"].values
    )
    NSBP_acc = accuracy_score(y, ypred)
    return NSP_acc, NSBP_acc


def compute_topk_accuracy(df, features, label_col, mode="NSB", k=5):
    X = StandardScaler().fit_transform(df[features].to_numpy())
    y = LabelEncoder().fit_transform(df[label_col].values)
    batches = df["batch"].values
    wells = df["well"].values
    # Get top-k neighbor labels
    knn_labels = evl.KNN_labels_NSBW(
        X, y, k, mode=mode, batches=batches, wells=wells, metric="cosine"
    )
    # Hit@k: any of the k neighbors matches the true label
    hits = (knn_labels == y[:, None]).any(axis=1)
    return hits.mean()


def compute_random_baseline_top1_NSB(df, features, label_col):
    # Shuffle labels and compute NSB top-1 accuracy
    X = StandardScaler().fit_transform(df[features].to_numpy())
    y = LabelEncoder().fit_transform(df[label_col].values)
    y_shuf = np.random.permutation(y)
    batches = df["batch"].values
    wells = df["well"].values
    ypred = evl.nearest_neighbor_classifier_NSBW(
        X, y_shuf, mode="NSB", batches=batches, wells=wells, metric="cosine"
    )
    return (ypred == y_shuf).mean()


def batch_aggregate(df, features):
    batch_prof = (
        df.groupby(["batch", "perturbation_id"])
        .agg({col: "mean" if col in features else "first" for col in df.columns})
        .reset_index(drop=True)
    )
    return batch_prof


def aggregate_consensus(df, features):
    cons_prof = (
        df.groupby("perturbation_id")
        .agg({col: "mean" if col in features else "first" for col in df.columns})
        .reset_index(drop=True)
    )
    return cons_prof


def compute_metrics(embed_list, indir, orf=False):
    print(f"Evaluating {len(embed_list)} embeddings: {embed_list}")

    # NSP and NSBP target / gene family accuracy
    target_label = "target" if not orf else "gene_group"

    dict_metrics = {}
    metric_names = [
        "label",
        "pert_acc",
        "pert_acc_NSB",
        f"{target_label}_acc_NSP",
        f"{target_label}_acc_NSBP",
        "pert_acc_NSB@5",
        "pert_acc_NSB@10",
        f"{target_label}_acc_NSBP@5",
        f"{target_label}_acc_NSBP@10",
        "pert_acc_NSB_random_baseline",
        "repl_corr",
        "percent_95",
    ]
    for i in metric_names:
        dict_metrics[i] = []
    pert_prec_recall_k = []
    pert_prec_recall_cor = []
    target_prec_recall_k = []
    target_prec_recall_cor = []

    for embed_idx, embed in enumerate(embed_list):
        print(f"Processing {embed_idx + 1}/{len(embed_list)}: {embed}")

        try:
            df, feature_type, features = load_embedding_data(indir, embed, orf)

            if len(df) == 0:
                print(f"Skipping {embed}: no data")
                continue

            # Log basic label cardinality
            print(
                f"Unique perturbations: {df['perturbation_id'].nunique()}  | Unique targets: {df[target_label].nunique()}"
            )

            # NN and NSB perturbation accuracy (top-1)
            pert_acc, NSB_pert_acc = compute_perturbation_accuracy(df, features)

            # NSP and NSBP target/gene family accuracy (top-1)
            NSP_target_acc, NSBP_target_acc = compute_target_accuracy(
                df, features, label=target_label
            )

            # Top-k retrieval (NSB for pert, NSBP for target)
            pert_acc_NSB_at5 = compute_topk_accuracy(
                df, features, label_col="perturbation_id", mode="NSB", k=5
            )
            pert_acc_NSB_at10 = compute_topk_accuracy(
                df, features, label_col="perturbation_id", mode="NSB", k=10
            )
            target_acc_NSBP_at5 = compute_topk_accuracy(
                df, features, label_col=target_label, mode="NSBW", k=5
            )
            target_acc_NSBP_at10 = compute_topk_accuracy(
                df, features, label_col=target_label, mode="NSBW", k=10
            )

            # Random baseline (top-1 NSB) for perturbation
            pert_nsb_rand = compute_random_baseline_top1_NSB(
                df, features, label_col="perturbation_id"
            )

            # percent replicating and mean replicate correlation
            repl_corr, _, percent_95, _ = evl.percent_replicating(
                df,
                n_samples=1000,
                n_replicates=12,
                replicate_grouping_feature="perturbation_id",
                feature_type=feature_type,
            )

            # batch-aggregated profiles
            batch_prof = batch_aggregate(df, features)

            # compute precision recall for perturbation ID classification
            pert_pr_k, pert_pr_cor = evl.calculate_precision_recall(
                batch_prof, features=features, label_col="perturbation_id"
            )
            pert_pr_k["label"] = embed
            pert_pr_cor["label"] = embed
            pert_prec_recall_k.append(pert_pr_k)
            pert_prec_recall_cor.append(pert_pr_cor)

            # consensus profiles
            cons_prof = aggregate_consensus(df, features)

            # precision-recall for target label matching on consensus profiles
            target_pr_k, target_pr_cor = evl.calculate_precision_recall(
                cons_prof, features=features, label_col=target_label
            )
            target_pr_k["label"] = embed
            target_pr_cor["label"] = embed
            target_prec_recall_k.append(target_pr_k)
            target_prec_recall_cor.append(target_pr_cor)

            # append well metrics to `dict_metrics`
            dict_metrics["label"].append(embed)
            dict_metrics["pert_acc"].append(pert_acc)
            dict_metrics["pert_acc_NSB"].append(NSB_pert_acc)
            dict_metrics[f"{target_label}_acc_NSP"].append(NSP_target_acc)
            dict_metrics[f"{target_label}_acc_NSBP"].append(NSBP_target_acc)
            dict_metrics["pert_acc_NSB@5"].append(pert_acc_NSB_at5)
            dict_metrics["pert_acc_NSB@10"].append(pert_acc_NSB_at10)
            dict_metrics[f"{target_label}_acc_NSBP@5"].append(target_acc_NSBP_at5)
            dict_metrics[f"{target_label}_acc_NSBP@10"].append(target_acc_NSBP_at10)
            dict_metrics["pert_acc_NSB_random_baseline"].append(pert_nsb_rand)
            dict_metrics["repl_corr"].append(np.nanmean(repl_corr))
            dict_metrics["percent_95"].append(percent_95)

        except Exception as e:
            print(f"❌ Error processing {embed}: {e}")
            continue

    if len(dict_metrics["label"]) == 0:
        print("❌ No metrics collected!")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    well_metrics = pd.DataFrame(dict_metrics)

    # Concatenate results
    pert_prec_recall_k_df = (
        pd.concat(pert_prec_recall_k, ignore_index=True)
        if pert_prec_recall_k
        else pd.DataFrame()
    )
    pert_prec_recall_cor_df = (
        pd.concat(pert_prec_recall_cor, ignore_index=True)
        if pert_prec_recall_cor
        else pd.DataFrame()
    )
    target_prec_recall_k_df = (
        pd.concat(target_prec_recall_k, ignore_index=True)
        if target_prec_recall_k
        else pd.DataFrame()
    )
    target_prec_recall_cor_df = (
        pd.concat(target_prec_recall_cor, ignore_index=True)
        if target_prec_recall_cor
        else pd.DataFrame()
    )

    return (
        well_metrics,
        pert_prec_recall_k_df,
        pert_prec_recall_cor_df,
        target_prec_recall_k_df,
        target_prec_recall_cor_df,
    )


def plot_metrics(
    prec_recall,
    well_metrics,
    figdir,
    xvar="pert",
    label_order=None,
    colorpal=["#f1bc41", "#901461", "#027e54", "#1C45E6", "#1CAFE6"],
):
    # Auto-detect label order from data if not provided
    if label_order is None:
        available_labels = prec_recall["label"].unique()
        # Use a default order but include all available labels
        default_order = [
            "DINO",
            "MAE",
            "SimCLR",
            "GRL-SimCLR",
            "CellProfiler",
            "transferlearning",
        ]
        label_order = [label for label in default_order if label in available_labels]
        # Add any remaining labels
        for label in available_labels:
            if label not in label_order:
                label_order.append(label)
    plt.figure(figsize=(6, 6))
    mAP_df = prec_recall[["label", "mAP"]].drop_duplicates(ignore_index=True)
    df_plot = pd.merge(mAP_df, well_metrics)
    xcol = f"{xvar}_acc_NSB" if xvar == "pert" else f"{xvar}_acc_NSBP"
    if df_plot["label"].str.contains("ViT").any():
        df_plot[["label", "architecture"]] = df_plot["label"].str.split(
            "-", n=1, expand=True
        )
        df_plot["architecture"] = df_plot["architecture"].replace(
            [None, "None"], "not applicable"
        )
        sn.scatterplot(
            data=df_plot,
            x=xcol,
            y="mAP",
            style="architecture",
            hue="label",
            hue_order=label_order,
            palette=colorpal,
            markers=["o", "^", "s"],
        )
    else:
        sn.scatterplot(
            data=df_plot,
            x=xcol,
            y="mAP",
            hue="label",
            hue_order=label_order,
            palette=colorpal,
        )
    sn.despine()
    if xvar == "target":
        label_type = xvar
    else:
        label_type = "perturbation" if xvar == "pert" else "gene family"
    nn_type = "NSB" if xvar == "pert" else "NSBP"
    plt.xlabel(f"{nn_type} {label_type} accuracy")
    plt.ylabel(f"mAP ({label_type})")
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc="upper left", frameon=False)
    plt.savefig(os.path.join(figdir, f"{xvar}_metrics_plot.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input directory with learned embeddings")
    parser.add_argument("-b", "--basedir", default="SSL_data/embeddings")
    parser.add_argument("--seed", default=507, type=int, help="Random seed")
    parser.add_argument(
        "--orf",
        action="store_true",
        help="Boolean flag indicating evaluation for ORF perturbations",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    basedir = args.basedir
    indir = os.path.join(basedir, args.input)

    if not os.path.exists(indir):
        print(f"❌ Directory not found: {indir}")
        exit(1)

    # create figure directory
    figdir = os.path.join(indir, "figures")
    os.makedirs(figdir, exist_ok=True)

    # list subdirectories in the input directory with embeddings
    embed_list_exp = [
        d
        for d in os.listdir(indir)
        if os.path.isdir(os.path.join(indir, d)) and d != "figures"
    ]

    if len(embed_list_exp) == 0:
        print("❌ No embeddings found!")
        exit(1)

    print(f"Found {len(embed_list_exp)} embeddings: {embed_list_exp}")

    # compute all metrics
    (
        well_metrics,
        pert_prec_recall_k,
        pert_prec_recall_cor,
        target_prec_recall_k,
        target_prec_recall_cor,
    ) = compute_metrics(embed_list_exp, indir, orf=args.orf)

    # save all metrics
    well_metrics.to_csv(os.path.join(indir, "well_metrics.csv"), index=False)
    pert_prec_recall_k.to_csv(
        os.path.join(indir, "pert_prec_recall_k.csv"), index=False
    )
    pert_prec_recall_cor.to_csv(
        os.path.join(indir, "pert_prec_recall_cor.csv"), index=False
    )
    target_prec_recall_k.to_csv(
        os.path.join(indir, "target_prec_recall_k.csv"), index=False
    )
    target_prec_recall_cor.to_csv(
        os.path.join(indir, "target_prec_recall_cor.csv"), index=False
    )

    print(f"✅ Saved evaluation metrics in {indir}")

    plot_metrics(pert_prec_recall_cor, well_metrics, figdir, xvar="pert")
    target_label = "target" if not args.orf else "gene_group"
    plot_metrics(target_prec_recall_cor, well_metrics, figdir, xvar=target_label)
    print(f"✅ Saved evaluation plots in {figdir}")
