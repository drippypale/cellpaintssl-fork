"""
Take a list of csv output evaluation metrics and plots features.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_pert_prec_recall_cor(indir, expdir):
    output_dir = os.path.join(indir, expdir)
    df = pd.read_csv(os.path.join(output_dir, 'pert_prec_recall_cor.csv'))
    df = df[['mAP', 'label']]
    df = df.drop_duplicates()
    return df

def load_target_prec_recall_cor(indir, expdir):
    output_dir = os.path.join(indir, expdir)
    df = pd.read_csv(os.path.join(output_dir, 'target_prec_recall_cor.csv'))
    df = df[['mAP', 'label']]
    df = df.drop_duplicates()
    return df

def load_pert_acc_NSB(indir, expdir):
    output_dir = os.path.join(indir, expdir)
    df = pd.read_csv(os.path.join(output_dir, 'well_metrics.csv'))
    df = df[['pert_acc_NSB', 'label']]
    return df

def load_target_acc_NSBP(indir, expdir):
    output_dir = os.path.join(indir, expdir)
    df = pd.read_csv(os.path.join(output_dir, 'well_metrics.csv'))
    df = df[['target_acc_NSBP', 'label']]
    return df

def plot_pert_acc_mAP (indir, expdir):
    df_acc= load_pert_acc_NSB(indir, expdir)
    df_mAP= load_pert_prec_recall_cor(indir, expdir)
    df = pd.merge(df_acc, df_mAP, on='label')
    sns.scatterplot(data=df, x='pert_acc_NSB', y='mAP', hue='label')
    

def plot_target_acc_mAP (indir, expdir):
    df_acc= load_target_acc_NSBP(indir, expdir)
    df_mAP= load_target_prec_recall_cor(indir, expdir)
    df = pd.merge(df_acc, df_mAP, on='label')
    sns.scatterplot(data=df, x='target_acc_NSBP', y='mAP', hue='label')

if __name__ == "__main__":
    indir = "embeddings directory"
    expdir = "output directory"

    plot_pert_acc_mAP (indir, expdir)  
    #plot_target_acc_mAP (indir, expdir)





    