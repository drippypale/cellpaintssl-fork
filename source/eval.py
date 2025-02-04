import pandas as pd
import numpy as np
import random
from scipy.stats import rankdata
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    average_precision_score,
    confusion_matrix,
    pairwise_distances,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from joblib import Parallel, delayed

def calculate_precision_recall(df, features,
                               label_col, 
                               dist_metric='correlation',
                               k_range = np.array(range(1,50,2)),
                               d_range =np.array(range(1,101,3))/100):

    knn_PR_curve = precision_recall_curve_knn(df, label_col, features,
                                                  dist_metric=dist_metric,
                                                  k_range=k_range)
    prec_recall_k = aggregate_PR_curve( knn_PR_curve, x_col='k' )
    prec_recall_k = prec_recall_k.fillna(0)
    prec_recall_k['AUCPR_k'] = np.trapz(x=prec_recall_k['avg_recall_micro'],
                                y=prec_recall_k['avg_precision_micro'])
    # Compute Precision recall curves based on distance thresholds
    dist_PR_curve = precision_recall_curve_dthresh(df, label_col, features,
                                                       dist_metric=dist_metric,
                                                       d_range=d_range)
    prec_recall_dist = aggregate_PR_curve( dist_PR_curve, x_col='d' )
    prec_recall_dist = prec_recall_dist.fillna(0)
    prec_recall_dist['AUCPR_dist'] = np.trapz(x=prec_recall_dist['avg_recall_micro'],
                                y=prec_recall_dist['avg_precision_micro'])
    return prec_recall_k, prec_recall_dist

def mean_aggregation(PR_curve, x_col):
    avg_PR_curve = (
        PR_curve.groupby(by=x_col)
        .mean()[["precision", "recall"]]
        .reset_index(drop=False)
    )
    return avg_PR_curve


def f1_from_PR(precision, recall):
    num = 2 * (precision * recall)
    denom = precision + recall
    f1 = np.divide(num, denom, out=np.zeros_like(num), where=(denom != 0))
    return f1


def aggregate_PR_curve(PR_curve, x_col):
    """Aggregates precission recall curves among all samples using different agregation functions (average, cumulative) and
    averaging types (micro, macro)"""

    # Compute average per label: for the macro aggregation versions
    PR_curve_per_label = PR_curve.groupby(by=["y_true", x_col]).mean().reset_index()

    # Average precision among samples (micro)
    name_map = {"precision": "avg_precision_micro", "recall": "avg_recall_micro"}
    avg_PR_curve_micro = mean_aggregation(PR_curve, x_col).rename(columns=name_map)
    avg_PR_curve_micro["avg_f1_micro"] = f1_from_PR(
        avg_PR_curve_micro["avg_precision_micro"].values,
        avg_PR_curve_micro["avg_recall_micro"].values,
    )

    # Average precision among labesl (macro)
    name_map = {"precision": "avg_precision_macro", "recall": "avg_recall_macro"}
    avg_PR_curve_macro = mean_aggregation(PR_curve_per_label, x_col).rename(
        columns=name_map
    )
    avg_PR_curve_macro["avg_f1_macro"] = f1_from_PR(
        avg_PR_curve_macro["avg_precision_macro"].values,
        avg_PR_curve_macro["avg_recall_macro"].values,
    )

    # Merge all aggregation results
    aggr_df = avg_PR_curve_micro.merge(avg_PR_curve_macro, on=x_col)
    if "mAP" in PR_curve.columns:
        aggr_df["mAP"] = PR_curve["mAP"].unique()[0]
    if "mAP_k" in PR_curve.columns:
        aggr_df["mAP_k"] = PR_curve["mAP_k"].unique()[0]
    if "mean_prec_R" in PR_curve.columns:
        aggr_df["mean_prec_R"] = PR_curve["mean_prec_R"].unique()[0]
    return aggr_df


def precision_at_R(y_true, knn_labels):

    # Encode labels as int32 to speedup computations and reduce memory for very large datasets
    encoder = preprocessing.LabelEncoder().fit(y_true)
    y_true = encoder.transform(y_true).astype(np.int32)
    knn_labels = (
        encoder.transform(knn_labels.flatten())
        .reshape(*knn_labels.shape)
        .astype(np.int32)
    )

    Rs = np.array([np.sum(y_true == y_true[i]) - 1 for i in range(len(y_true))])
    TP_at_R = np.array(
        [
            np.sum(np.repeat(y_true[i], R) == knn_labels[i, :R])
            for R, i in zip(Rs, range(len(Rs)))
        ]
    )
    prec_at_R = TP_at_R / Rs
    return np.nanmean(prec_at_R)


from time import time


def precision_recall_curve_knn(
    data,
    label_col,
    feature_cols,
    k_range=np.array(range(1, 50, 2), dtype=int),
    dist_metric="correlation",
    n_jobs=8,
    dist_mat=np.array([]),
):
    """Preciscion and recall curves per sample over a range of k-values"""

    assert dist_metric in [
        "correlation",
        "cosine",
    ], "Invalid metric. Only bounded metrics are supported"

    # Get nearest neighbors
    X, y_true = data[feature_cols].values, data[label_col].values
    knn_labels = KNN_labels_NSBW(
        X, y_true, k_range[-1], metric=dist_metric, dist_mat=dist_mat
    )

    # Compute metrics for different numbers of neighbors
    PR_curve = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(precision_recall_from_knns)(y_true, knn_labels, k) for k in k_range
    )
    PR_curve = pd.concat(PR_curve).reset_index(drop=True)

    t0 = time()
    PR_curve["mean_prec_R"] = precision_at_R(y_true, knn_labels)
    print("Precission at R", time() - t0)
    print("mean", PR_curve["mean_prec_R"].mean())

    return PR_curve


def precision_recall_curve_dthresh(
    data,
    label_col,
    feature_cols,
    d_range=np.array(range(1, 100, 4)) / 100,
    dist_metric="correlation",
    n_jobs=8,
    dist_mat=np.array([]),
):
    """Preciscion and recall curves per sample over a range of distance-values"""

    assert dist_metric in [
        "correlation",
        "cosine",
    ], "Invalid metric. Only bounded metrics are supported"

    # Compute distance matrix
    X, y_true = data[feature_cols].values, data[label_col].values
    if dist_mat.size == 0:
        dist_mat = pairwise_distances_parallel(X, metric=dist_metric)
    np.fill_diagonal(dist_mat, dist_mat.max())

    # Compute metrics for different distance thresholds
    PR_curve = pd.DataFrame()
    PR_curve = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(precision_recall_from_dist_mat)(y_true, dist_mat, d) for d in d_range
    )
    PR_curve = pd.concat(PR_curve)
    PR_curve["mAP"] = mean_average_precision(y_true, dist_mat)
    PR_curve["mAP_k"] = mean_average_precision(y_true, dist_mat, mode='k')

    return PR_curve


def precision_recall_from_knns(y_true, knn_labels, k):
    """Precission and recall for a given number of nearest neighbors"""

    # Pick top k
    knn_labels = knn_labels[:, 0:k]

    # Encode labels as int32 to speedup computations and reduce memory for very large datasets
    encoder = preprocessing.LabelEncoder().fit(y_true)
    y_true = encoder.transform(y_true).astype(np.int32)
    knn_labels = (
        encoder.transform(knn_labels.flatten())
        .reshape(*knn_labels.shape)
        .astype(np.int32)
    )

    # True Positives: Number of times the correct label is in the k-nearest neighbors labels
    y_true_exp = np.repeat(y_true[:, None], k, axis=1)
    TP = (y_true_exp == knn_labels).sum(axis=1)
    df = pd.DataFrame({"y_true": y_true, "TP": TP})

    # Number of samples per each true label
    counts = df["y_true"].value_counts()
    counts = pd.DataFrame({"y_true": counts.index, "y_true_count": counts.values})
    df = df.merge(counts, how="left")

    # Precision and recall
    df["precision"] = df["TP"] / k
    df["recall"] = df["TP"] / (df["y_true_count"] - 1)

    # final formating
    df = df.drop(columns=["TP"])
    df["k"] = k

    return df


def precision_recall_from_dist_mat(y_true, dist_mat, dist_thresh):
    """Precission and recall for a given distance threshold"""

    # Encode labels as int32 to speedup computations and reduce memory for very large datasets
    y_true = preprocessing.LabelEncoder().fit_transform(y_true).astype(np.int32)

    # True Positives: Number of times the correct label is in the k-nearest neighbors labels
    y_true_exp = np.repeat(y_true[:, None], dist_mat.shape[1], axis=1)
    y_pred_exp = np.repeat(y_true[None, :], dist_mat.shape[0], axis=0)

    # Asign default wrong label to all y_pred above the distance threshold
    default_wrong_label = y_true.max() + 1
    is_in_neighborhood = dist_mat < (dist_thresh + 1e-6)
    y_pred_exp[is_in_neighborhood == False] = default_wrong_label

    # True positives for each row
    TP = (y_true_exp == y_pred_exp).sum(axis=1)
    num_neighbors = is_in_neighborhood.sum(axis=1)
    df = pd.DataFrame({"y_true": y_true, "TP": TP, "num_neighbors": num_neighbors})

    # Number of samples per each true label
    counts = df["y_true"].value_counts()
    counts = pd.DataFrame({"y_true": counts.index, "y_true_count": counts.values})
    df = df.merge(counts, how="left")

    # Precision and recall
    df["precision"] = df["TP"] / df["num_neighbors"]
    df["recall"] = df["TP"] / (df["y_true_count"] - 1)

    # final formating
    df = df.drop(columns=["TP"])
    df["d"] = dist_thresh

    return df


def mean_average_precision(y_true, dist_mat, mode='d'):
    # Encode labels as int32 to speedup computations and reduce memory for very large datasets
    y_true = preprocessing.LabelEncoder().fit_transform(y_true).astype(np.int32)

    # True Positives: Number of times the correct label is in the k-nearest neighbors labels
    y_true_exp = np.repeat(y_true[:, None], dist_mat.shape[1], axis=1)

    # mean average precision (mAP)
    y_gt = np.repeat(y_true[None, :], dist_mat.shape[0], axis=0)
    y_true_ap = (y_gt == y_true_exp).astype(np.float32)
    np.fill_diagonal(y_true_ap, np.nan)
    np.fill_diagonal(dist_mat, np.nan)
    if mode == 'd':
        ap = [
            average_precision_score(
                y_true=remove_nan(y_true_ap[i]).astype(np.int32),
                y_score=1. - remove_nan(dist_mat[i]),
            )
            for i in range(dist_mat.shape[0])
        ]
    if mode == 'k':
        ap = [
            average_precision_score(
                # remove_nan removes masked values (on the diagonal, i.e. matchng to itself)
                y_true=remove_nan(y_true_ap[i]).astype(np.int32),
                # y_score is a ranked similarity
                y_score= rankdata(1. - remove_nan(dist_mat[i])),
            )
            for i in range(dist_mat.shape[0])
        ]

    return np.nanmean(ap)


def KNN_labels_NSBW(
    X, y, k, mode="NN", batches=[], wells=[], metric="cosine", dist_mat=np.array([])
):
    """
    Get the label of the k nearest neighbors for each sample in X but, optionally, excluding all samples which belong to the same batch or the same well.
    The code first uses a partition on the distance matrix to retreive unsorted k-nearest neighbors. Then it performs sorting among the K neighbors.
    This makes the code rather unreadable but it is much faster than performing a full sort, especially whne K << num samples
    :param X: feature matrix (shape = samples, features)
    :param y: labels vector (len = samples )
    :param mode: Type of calssifier taks
        - NN: simple nearest neighbor conisdering all samples in the data
        - NSB: not same bacth
        - NSW: not same well
        - NSBW: not same batch or well
    :param batches: vector with batch assignments (len = samples )
    :param wells: vector with well assignments (len = samples )
    :param dist_mat: precomputed distance matrix
    """
    assert mode in ["NN", "NSB", "NSW", "NSBW"], "unknown mode"

    if dist_mat.size == 0:
        dist_mat = pairwise_distances_parallel(X, metric=metric)
    np.fill_diagonal(dist_mat, dist_mat.max())

    # For each compound, penilize compounds from the same batch and/or well
    # by assigning them the maximum distance
    max_dist = dist_mat.max()
    if mode != "NN":
        for cpd_idx in range(X.shape[0]):
            if "B" in mode:
                same_batch_idx = batches == batches[cpd_idx]
                dist_mat[cpd_idx, same_batch_idx] = max_dist
            if "W" in mode:
                same_well_idx = wells == wells[cpd_idx]
                dist_mat[cpd_idx, same_well_idx] = max_dist

    # Use partition to return UNORDERED indices from the K smalles values for each row
    col_indices = np.argpartition(dist_mat, k)[:, :k]

    # Sort the K selected column indexes for each row
    rows = np.repeat(np.arange(dist_mat.shape[0]), k)
    values = dist_mat[rows, col_indices.ravel()].reshape(dist_mat.shape[0], k)
    sorted_col_indices_idx = np.argsort(values, axis=1)
    sorted_col_indices = col_indices[rows, sorted_col_indices_idx.ravel()].reshape(
        dist_mat.shape[0], k
    )

    knn_labels = y[sorted_col_indices]

    return knn_labels


def nearest_neighbor_classifier_NSBW(
    X, y, mode="NN", batches=[], wells=[], metric="cosine"
):
    """
    Get the label of its nearest neighbor for each sample in X but, optionally, excluding all samples which belong to the same batch or the same well
    :param X: feature matrix (shape = samples, features)
    :param y: labels vector (len = samples )
    :param mode: Type of calssifier taks
        - NN: simple nearest neighbor conisdering all samples in the data
        - NSB: not same bacth
        - NSW: not same well
        - NSBW: not same batch or well
    :param batches: vector with batch assignments (len = samples )
    :param wells: vector with well assignments (len = samples )
    """
    assert mode in ["NN", "NSB", "NSW", "NSBW"], "unknown mode"

    dist_mat = pairwise_distances_parallel(X, metric=metric)
    max_dist = dist_mat.max()
    np.fill_diagonal(dist_mat, max_dist)

    # For each compound, penilize compounds from the same batch and/or well
    # by assigning them the maximum distance
    if mode != "NN":
        for cpd_idx in range(X.shape[0]):
            if "B" in mode:
                same_batch_idx = np.logical_and(batches == batches[cpd_idx], y == y[cpd_idx])
                dist_mat[cpd_idx, same_batch_idx] = max_dist
            if "W" in mode:
                same_well_idx = wells == wells[cpd_idx]
                dist_mat[cpd_idx, same_well_idx] = max_dist

    knn_idxs = np.argmin(dist_mat, axis=1)
    y_pred = y[knn_idxs]

    return y_pred


def pairwise_distances_parallel(
    X, metric="euclidean", n_jobs=8, min_samples=15000, chunck_size=5000
):
    """Splits computation of distance matrix into several chuncks for very large only"""

    if X.shape[0] >= min_samples:
        max_i = int(np.ceil(X.shape[0] / chunck_size))
        chunk_idxs = [
            [i * chunck_size, min(X.shape[0], (i + 1) * chunck_size)]
            for i in range(max_i)
        ]
        dist_mat = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(pairwise_distances)(X[idx[0] : idx[1], :], X, metric)
            for idx in chunk_idxs
        )
        dist_mat = np.concatenate(dist_mat)
    else:
        dist_mat = pairwise_distances(X, metric=metric)

    return dist_mat


def percent_replicating(
    df, n_samples, n_replicates, replicate_grouping_feature, feature_type="cellprofiler"
):
    """Wraps steps to calculate all inputs for the replicate correlation plot"""
    replicate_corr, replicate_names = corr_between_replicates(
        df, replicate_grouping_feature, feature_type=feature_type
    )
    sort_idx = np.argsort(replicate_corr)
    replicate_corr, replicate_names = (
        replicate_corr[sort_idx],
        replicate_names[sort_idx],
    )
    null_replicate_corr = list(
        corr_between_non_replicates(
            df,
            n_samples=n_samples,
            n_replicates=n_replicates,
            metadata_compound_name=replicate_grouping_feature,
            feature_type=feature_type,
        )
    )
    percent_95, thresh_95 = percent_score(
        null_replicate_corr, replicate_corr, how="right"
    )
    return replicate_corr, null_replicate_corr, percent_95, thresh_95


def percent_score(null_dist, corr_dist, how):
    """
    Calculates the Percent strong or percent recall scores
    :param null_dist: Null distribution
    :param corr_dist: Correlation distribution
    :param how: "left", "right" or "both" for using the 5th percentile, 95th percentile or both thresholds
    :return: proportion of correlation distribution beyond the threshold
    """
    if how == "right":
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        return np.mean(above_threshold.astype(float)), perc_95
    if how == "left":
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return np.mean(below_threshold.astype(float)), perc_5
    if how == "both":
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return (
            np.mean(above_threshold.astype(float))
            + np.mean(below_threshold.astype(float)),
            perc_95,
            perc_5,
        )


def corr_between_replicates(df, group_by_feature, feature_type="cellprofiler"):
    """
    Correlation between replicates
    :param df: pd.DataFrame
    :param group_by_feature: Feature name to group the data frame by
    :return: list-like of correlation values
    """
    replicate_corr = []
    replicate_grouped = df.groupby(group_by_feature)
    group_names = []
    for name, group in replicate_grouped:
        group_features = get_feature_data(group, feature_type=feature_type)
        corr = np.corrcoef(group_features)
        if len(group_features) == 1:  # If there is only one replicate on a plate
            replicate_corr.append(np.nan)
        else:
            np.fill_diagonal(corr, np.nan)
            replicate_corr.append(np.nanmedian(corr))  # median replicate correlation
        group_names.append(name)
    return np.array(replicate_corr), np.array(group_names)


def corr_between_non_replicates(
    df, n_samples, n_replicates, metadata_compound_name, feature_type="cellprofiler"
):
    """
    Null distribution between random "replicates".
    :param df: pandas.DataFrame
    :param n_samples: int
    :param n_replicates: int
    :param metadata_compound_name: Compound name feature
    :return: list-like of correlation values, with a  length of `n_samples`
    """
    df.reset_index(drop=True, inplace=True)
    null_corr = []
    while len(null_corr) < n_samples:
        compounds = random.choices([_ for _ in range(len(df))], k=n_replicates)
        sample = df.loc[compounds].copy()
        if len(sample[metadata_compound_name].unique()) == n_replicates:
            sample_features = get_feature_data(sample, feature_type=feature_type)
            corr = np.corrcoef(sample_features)
            np.fill_diagonal(corr, np.nan)
            null_corr.append(np.nanmedian(corr))  # median replicate correlation
    return null_corr


def corr_between_perturbation_pairs(df, metadata_common, metadata_perturbation):
    """
    Correlation between perturbation pairs
    :param df: pd.DataFrame
    :param metadata_common: feature that identifies perturbation pairs
    :param metadata_perturbation: perturbation name feature
    :return: list-like of correlation values
    """
    replicate_corr = []

    profile_df = get_metadata(df).assign(profiles=list(get_featuredata(df).values))

    replicate_grouped = (
        profile_df.groupby([metadata_common, metadata_perturbation])
        .profiles.apply(list)
        .reset_index()
    )

    common_grouped = (
        replicate_grouped.groupby([metadata_common]).profiles.apply(list).reset_index()
    )

    for i in range(len(common_grouped)):
        if len(common_grouped.iloc[i].profiles) > 1:
            compound1_profiles = common_grouped.iloc[i].profiles[0]
            compound2_profiles = common_grouped.iloc[i].profiles[1]

            corr = np.corrcoef(compound1_profiles, compound2_profiles)
            corr = corr[
                0 : len(common_grouped.iloc[i].profiles[0]),
                len(common_grouped.iloc[i].profiles[0]) :,
            ]
            replicate_corr.append(np.nanmedian(corr))

    return replicate_corr


def corr_between_perturbation_non_pairs(
    df, n_samples, metadata_common, metadata_perturbation
):
    """
    Null distribution generated by computing correlation between random pairs of perturbations.
    :param df: pandas.DataFrame
    :param n_samples: int
    :param metadata_common: feature that identifies perturbation pairs
    :param metadata_perturbation: metadata_perturbation: perturbation name feature
    :return: list-like of correlation values, with a  length of `n_samples`
    """
    df.reset_index(drop=True, inplace=True)
    null_corr = []

    profile_df = get_metadata(df).assign(profiles=list(get_featuredata(df).values))

    replicate_grouped = (
        profile_df.groupby([metadata_common, metadata_perturbation])
        .profiles.apply(list)
        .reset_index()
    )

    while len(null_corr) < n_samples:
        compounds = random.choices([_ for _ in range(len(replicate_grouped))], k=2)
        compound1_moa = replicate_grouped.iloc[compounds[0]][metadata_common]
        compound2_moa = replicate_grouped.iloc[compounds[1]][metadata_common]
        if compound1_moa != compound2_moa:
            compound1_profiles = replicate_grouped.iloc[compounds[0]].profiles
            compound2_profiles = replicate_grouped.iloc[compounds[1]].profiles
            corr = np.corrcoef(compound1_profiles, compound2_profiles)
            corr = corr[
                0 : len(replicate_grouped.iloc[0].profiles),
                len(replicate_grouped.iloc[0].profiles) :,
            ]
            null_corr.append(np.nanmedian(corr))  # median replicate correlation
    return null_corr


def embeddings_preprocessing(df, grouping_feature="batch"):
    """
    Helper function to preprocesses cell painting embeddings
    :param df: pd.DataFrame
    :grouping_feature str: name of the batch variable in df
    :return: preprocessed pd.DataFrame
    """
    # CellProfiler features start with either 'C' or 'N', deep learning features with 'emb'
    df = df.filter(regex=r"^emb|^[C, N]|batch").copy()
    batch_labels = df.loc[:, grouping_feature].astype("category").cat.codes.values
    df.loc[:, "batch_labels"] = batch_labels
    return df


def batch_classification(df, train_size=0.8, random_state=100):
    """
    Fit a RF classifier to predict the batch based on the embeddings.
    :param df: pd.DataFrame
    :param train_size: float
    :return: dictionary with keys 'MCC', 'ROC_AUC', 'confusion_matrix' and Gini 'feature_importance' sorted by importance
    """
    # CellProfiler features start with either 'C' or 'N', deep learning features with 'emb'
    df = embeddings_preprocessing(df)

    df_train, df_test = train_test_split(
        df, train_size=train_size, stratify=df.batch_labels, random_state=random_state
    )
    labels_train = df_train.batch_labels
    labels_test = df_test.batch_labels
    features_regex = r"^emb|^[C, N]"
    features_train = df_train.filter(regex=features_regex)
    features_test = df_test.filter(regex=features_regex)

    classifier = RandomForestClassifier(n_estimators=200)
    classifier.fit(features_train, labels_train)
    labels_pred_score = classifier.predict_proba(features_test)
    labels_pred = labels_pred_score.argmax(axis=1)

    feature_importance = pd.DataFrame(
        {
            "feature": features_train.columns,
            "importance": classifier.feature_importances_,
        },
        index=None,
    ).sort_values("importance", ascending=False, ignore_index=True)

    return {
        "MCC": matthews_corrcoef(labels_test, labels_pred),
        "ROC_AUC": roc_auc_score(labels_test, labels_pred_score, multi_class="ovr"),
        "confusion_matrix": confusion_matrix(labels_test, labels_pred),
        "feature_importance": feature_importance,
    }


def batch_clustering(df, random_state=100, method="KMeans"):
    """
    Fit a clustering algorithm with the number of batches and run clustering diagnostics
    :param df: pd.DataFrame
    :random_state int: seed for K-Means
    :method str: choose between 'KMeans' and 'Agglomerative'
    :return dict: dictionary with 'ARI', 'AMI', and 'Silhouette'. The latter is computed with the true labels!
    """

    df = embeddings_preprocessing(df)
    features = df.filter(regex=r"^emb|^[C, N]")
    n_clusters = df.batch_labels.nunique()

    if method == "KMeans":
        clustering = KMeans(n_clusters=n_clusters, random_state=random_state)
    elif method == "Agglomerative":
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(
            f"{method} not supported: Please choose 'KMeans' or 'Agglomerative'"
        )
    print(f"Clustering with {method}...")
    labels_clustering = clustering.fit_predict(features)
    labels_true = df.batch_labels

    return {
        "ARI": adjusted_rand_score(labels_true, labels_clustering),
        "AMI": adjusted_mutual_info_score(labels_true, labels_clustering),
        "Silhouette": silhouette_score(features, labels_true),
    }


# -------------------------- From here fucntions are adapted to fit our workflow to Niranj's analysis ----------
# Code complied from Niranj's Scripts used to analyze the JUMP-CP pilot 1 experiment Adapted from: https://github.com/jump-cellpainting/workflow_demo_analysis/blob/main/analysis_Broad/0.percent_scores.ipynb


def get_feature_cols(df, feature_type="cellprofiler"):
    """Splits columns of input dataframe into columns contining metadata and columns containing morphological profile features
    :param df: input data frame
    :param features_type:
        "standard" for features named as feature_1, feature_2 ..
        "CellProfiler": for cell-centric Cell Profiler features names as Cells_ , Nuclei_ , Cytoplasm_
    :return : feature_columns , info_columns
    """
    if feature_type.lower() == "cellprofiler":
        feature_cols = [
            c
            for c in df.columns
            if (
                c.startswith("Cells_")
                | c.startswith("Nuclei_")
                | c.startswith("Cytoplasm_")
                | c.startswith("Image_")
            )
            & ("Metadata" not in c)
        ]
    elif feature_type == "standard":
        feature_cols = [
            c for c in df.columns if (c.startswith("feature_") | c.startswith("emb"))
        ]
    else:
        raise NotImplementedError(
            "Feature Type not implemented. Options: CellProfiler, standard"
        )
    info_cols = [c for c in df.columns if not (c in feature_cols)]
    return feature_cols, info_cols


def get_feature_data(df, feature_type="cellprofiler"):
    """return dataframe of just feature columns.
    This is a bridging function to use Niranj scripts since he uses teh key word Metadata to define metadata columns which does not hold for us"""
    feature_cols, _ = get_feature_cols(df, feature_type)
    return df[feature_cols]


def get_metadata(df, features_type="cellprofiler"):
    """return dataframe of just metadata columns.
    This is a bridging function to use Niranj scripts since he uses teh key word Metadata to define metadata columns which does not hold for us"""
    _, meta_cols = get_feature_cols(df, features_type)
    return df[meta_cols]


def remove_nan(x):
    return x[~np.isnan(x)]
