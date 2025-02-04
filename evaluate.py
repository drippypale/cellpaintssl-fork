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
    df = pd.read_csv(os.path.join(embed_dir, 'well_features.csv'))
    # remove DMSO wells
    df = df.query('perturbation_id != "DMSO"')
    if orf:
        df = df.query('control_type != "negcon"')
    feature_type = 'cellprofiler' if embed in ['CellProfiler', 'Random', 'Shuffle CP'] else 'standard'
    features, _ = utils.get_feature_cols(df, feature_type=feature_type)
    return df, feature_type, features

def compute_perturbation_accuracy(df, features):
    X = StandardScaler().fit_transform(df[features].to_numpy())
    y = LabelEncoder().fit_transform(df['perturbation_id'].values)
    # Simple perturbation accuracy
    ypred = evl.nearest_neighbor_classifier_NSBW(X, y, mode='NN', metric='cosine')
    acc = accuracy_score(y,  ypred)
    # NSB perturbation accuracy
    batches = df['batch'].values
    ypred = evl.nearest_neighbor_classifier_NSBW(X, y, mode='NSB', batches=batches, metric='cosine')
    NSB_acc = accuracy_score(y,  ypred)
    return acc, NSB_acc

def compute_target_accuracy(df, features, label='target'):
    X = StandardScaler().fit_transform(df[features].to_numpy())
    y = LabelEncoder().fit_transform(df[label].values)
    # not-same-perturbation (NSP) target accuracy
    ypred = evl.nearest_neighbor_classifier_NSBW(X, y, mode='NSW', metric='cosine', wells=df['well'].values)
    NSP_acc = accuracy_score(y, ypred)
    # not-same-batch-or-perturbation (NSBP) target accuracy
    batches = df['batch'].values
    ypred = evl.nearest_neighbor_classifier_NSBW(X, y, mode='NSBW', batches=batches, metric='cosine', wells=df['well'].values)
    NSBP_acc = accuracy_score(y,  ypred)
    return NSP_acc, NSBP_acc

def batch_aggregate(df, features):
    batch_prof = (df.groupby(['batch', 'perturbation_id']).
                        agg({col: 'mean' if col in features else 'first' for col in df.columns}).
                        reset_index(drop=True))
    return batch_prof

def aggregate_consensus(df, features):
    cons_prof = (df.groupby('perturbation_id').
                        agg({col: 'mean' if col in features else 'first' for col in df.columns}).
                        reset_index(drop=True))
    return cons_prof

def compute_metrics(embed_list, indir, orf=False):
    # NSP and NSBP target / gene family accuracy
    target_label = 'target' if not orf else 'gene_group'
    dict_metrics = {}
    metric_names = ["label", "pert_acc", "pert_acc_NSB", \
        f"{target_label}_acc_NSP", f"{target_label}_acc_NSBP", \
        "repl_corr", "percent_95"]
    for i in metric_names:
        dict_metrics[i] = []
    pert_prec_recall_k = []
    pert_prec_recall_cor = []
    target_prec_recall_k  = []
    target_prec_recall_cor = []

    for embed in embed_list:
        print(f"Computing metrics on well profiles for {embed}")
        df, feature_type, features = load_embedding_data(indir, embed, orf)
        # NN and NSB perturbation accruacy
        pert_acc, NSB_pert_acc = compute_perturbation_accuracy(df, features)
        # NSP and NSBP target/gene family accuracy
        NSP_target_acc, NSBP_target_acc = compute_target_accuracy(df, features, label=target_label)
        # percent replicating and mean replicate correlation
        repl_corr, _, percent_95, _ = evl.percent_replicating(df,
                                                                        n_samples=1000,
                                                                        n_replicates=12,
                                                                        replicate_grouping_feature='perturbation_id',
                                                                        feature_type=feature_type)
        # batch-aggregated profiles
        print(f"Computing batch-aggregated metrics for {embed}")
        batch_prof = batch_aggregate(df, features)
        # compute precision recall for perturbation ID classification
        pert_pr_k, pert_pr_cor = evl.calculate_precision_recall(batch_prof, features=features,
                                                                    label_col = 'perturbation_id')
        pert_pr_k['label'] = embed
        pert_pr_cor['label'] = embed
        pert_prec_recall_k.append(pert_pr_k)
        pert_prec_recall_cor.append(pert_pr_cor)
        
        print(f"Computing consensus profile metrics for {embed}")
         # consensus profiles
        cons_prof = aggregate_consensus(df, features)
        # precision-recall for target label matching on consensus profiles
        target_pr_k, target_pr_cor = evl.calculate_precision_recall(cons_prof, features=features,
                                                                    label_col=target_label)
        target_pr_k['label'] = embed
        target_pr_cor['label'] = embed
        target_prec_recall_k.append(target_pr_k)
        target_prec_recall_cor.append(target_pr_cor)
        # append well metrics to `dict_metrics`
        dict_metrics["label"].append(embed)
        dict_metrics["pert_acc"].append(pert_acc)
        dict_metrics["pert_acc_NSB"].append(NSB_pert_acc)
        dict_metrics[f"{target_label}_acc_NSP"].append(NSP_target_acc)
        dict_metrics[f"{target_label}_acc_NSBP"].append(NSBP_target_acc)
        dict_metrics["repl_corr"].append(np.nanmean(repl_corr))
        dict_metrics["percent_95"].append(percent_95)
    well_metrics = pd.DataFrame(dict_metrics)
    return well_metrics, pd.concat(pert_prec_recall_k, ignore_index=True), \
           pd.concat(pert_prec_recall_cor, ignore_index=True), \
           pd.concat(target_prec_recall_k, ignore_index=True), \
           pd.concat(target_prec_recall_cor, ignore_index=True)

def plot_metrics(prec_recall, well_metrics, figdir, xvar='pert',
                              label_order=['DINO', 'MAE', 'SimCLR',
                                           'CellProfiler', 'transferlearning'],
                              colorpal = ['#f1bc41', '#901461', '#027e54',
                                          '#1C45E6', '#1CAFE6']):
    plt.figure(figsize=(6,6))
    mAP_df = prec_recall[['label', 'mAP']].drop_duplicates(ignore_index=True)
    df_plot = pd.merge(mAP_df, well_metrics)
    xcol = f'{xvar}_acc_NSB' if xvar == 'pert' else f'{xvar}_acc_NSBP'
    if df_plot['label'].str.contains('ViT').any():
        df_plot[['label', 'architecture']] = df_plot['label'].str.split('-', n=1, expand=True)
        df_plot['architecture'] = df_plot['architecture'].replace([None, 'None'],'not applicable')
        sn.scatterplot(data=df_plot, x=xcol, y='mAP', style='architecture',
               hue='label', hue_order=label_order, palette=colorpal,
               markers=['o', '^', 's'])
    else:
        sn.scatterplot(data=df_plot, x=xcol, y='mAP',
                    hue='label', hue_order=label_order, palette=colorpal)
    sn.despine()
    if xvar == 'target':
        label_type = xvar
    else:
        label_type = 'perturbation' if xvar == 'pert' else 'gene family'
    nn_type = 'NSB' if xvar == 'pert' else 'NSBP'
    plt.xlabel(f'{nn_type} {label_type} accuracy')
    plt.ylabel(f'mAP ({label_type})')
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', frameon=False)
    plt.savefig(os.path.join(figdir, f'{xvar}_metrics_plot.pdf'), bbox_inches='tight')

if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input directory with learned embeddings')
    parser.add_argument('-b', '--basedir', default='SSL_data/embeddings')
    parser.add_argument('--seed', default=507, type=int, help='Random seed')
    parser.add_argument('--orf', action='store_true', help='Boolean flag indicating evaluation for ORF perturbations')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    basedir = args.basedir 
    # directory of trained embeddings
    indir = os.path.join(basedir, args.input)
    print(f"Reading embeddings from: {indir}")

    # create figure directory
    figdir = os.path.join(indir, 'figures')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    # list subdirectories in the input directory with embeddings
    embed_list_exp = [d for d in os.listdir(indir) if os.path.isdir(os.path.join(indir, d)) and d != 'figures']
    print("Embeddings to evaluate: ", embed_list_exp, "\n")
    # compute all metrics
    well_metrics, pert_prec_recall_k, \
         pert_prec_recall_cor, \
         target_prec_recall_k, \
         target_prec_recall_cor = compute_metrics(embed_list_exp, indir, orf=args.orf)
    # save all metrics
    well_metrics.to_csv(os.path.join(indir, 'well_metrics.csv'), index=False)
    pert_prec_recall_k.to_csv(os.path.join(indir, 'pert_prec_recall_k.csv'), index=False)
    pert_prec_recall_cor.to_csv(os.path.join(indir, 'pert_prec_recall_cor.csv'), index=False)
    target_prec_recall_k.to_csv(os.path.join(indir, 'target_prec_recall_k.csv'), index=False)
    target_prec_recall_cor.to_csv(os.path.join(indir, 'target_prec_recall_cor.csv'), index=False)
    print(f"Saved evaluation metrics in {indir}")

    plot_metrics(pert_prec_recall_cor, well_metrics, figdir, xvar='pert')
    target_label = 'target' if not args.orf else 'gene_group'
    plot_metrics(target_prec_recall_cor, well_metrics, figdir, xvar=target_label)
    print(f"Saved evaluation plots in {figdir}")
    
