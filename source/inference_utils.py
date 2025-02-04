import torch
import pandas as pd
import numpy as np
import pycytominer
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from source.utils import get_feature_cols


def forward_inference(embedding_net, batch_of_crops, labels, device:str):#, agg_method='median'):
    # crops dim is (batch_size, n_crops, C, W, H)
    # reshape to (batch_size*n_crops, C, W, H) for GPU support
    # labels: tensor of 0 and 1 indicating whether the crop had enough cells: shape = (B, Ncrops)
    with torch.no_grad():
        #agg_methods = {'median': torch.nanmedian}#, 'mean': torch.nanmean}
        assert len(batch_of_crops.shape) == 5
        B, Ncrops, C, W, H = batch_of_crops.shape
        batch_of_crops = torch.reshape(batch_of_crops, (B*Ncrops, C, W, H))
        batch_of_crops = batch_of_crops.to(device)
        batch_of_embeddings = embedding_net(batch_of_crops)
        batch_of_embeddings = torch.reshape(batch_of_embeddings, (B, Ncrops, -1))
        batch_of_embeddings[~labels] = np.nan
    return batch_of_embeddings

def forward_inference_wholeimg(embedding_net, fov, device:str):
    # whole-image inference for ViT-based models
    with torch.no_grad():
        assert len(fov.shape) == 4
        fov = fov.to(device)
        batch_of_embeddings = embedding_net(fov)
    return batch_of_embeddings

def aggregate_embeddings_plate(plate_dfr, plate_embs,
                               my_cols = ['batch', 'plate', 
                                          'well', 'perturbation_id',
                                          'target'],  operation="mean"):
    """
    This function takes a data frame and a list containing embeddings as it is produced in 03_inference.py
    The plate_dfr should be a subset of train_df containing all the metadata for one plate.
    The plate_embs is a list where each list element is the result of the forward pass of one minibatch, i.e. it is a list with each element being the result of one minibatch. Each element is of shape (minibatch_size, ncrops, embedding_size)
    """
    assert operation in ["median", 'mean'] , f"Specify aggregation function {operation} received while only 'median' and 'mean' are supported"
    # Ensure we have as many embeddings as rows in the data frame: 
    # Note that the last minibatch size may be different from the other minibatch sizes. 
    assert(len(plate_dfr) == ((len(plate_embs)-1) * len(plate_embs[0])) + len(plate_embs[-1]))

    # this here converts all of these arrays into a list in which each element being one embedding
    #print("Splitting plate_embs")
    plate_embs_split = [np.split(emb, indices_or_sections=emb.shape[0], axis=0) for emb in plate_embs]

    # convert into a list with one element per row of plate_dfr 
    # each element of that list has shape = (Ncrops, embedding_size)
    plate_embs_flat = [item.squeeze() for sublist in plate_embs_split for item in sublist]

    list_of_dicts = []
    for r in range(len(plate_dfr)):
        d = {}
        d['data'] = plate_embs_flat[r]
        d['identifier'] = "_".join(str(el) for el in plate_dfr.loc[r, my_cols])
        list_of_dicts.append(d)

    # plate_dfr_smry has one row for each well in the given plate, i.e. the embeddings over 
    # all fields of view will have been aggregated
    plate_dfr_smry = plate_dfr.loc[:, my_cols].drop_duplicates().reset_index(drop=True)
    len(plate_dfr_smry)

    #print("Collecting means ...")
    list_of_median_embs = []
    for r in range(len(plate_dfr_smry)):
        to_look_for = "_".join(str(el) for el in plate_dfr_smry.loc[r, my_cols])
        collected = []
        matched = []
        for el in list_of_dicts:
            if el['identifier'] == to_look_for:
                collected.append(el['data'])
                matched.append(True)
            else:
                matched.append(False)
        # shrink list, kicking out everything that was not a match
        list_of_dicts = [el for (el, is_match) in zip(list_of_dicts, matched) if not is_match]
        # perform numpy computation, then add to list_of_median_embs
        tmp = np.vstack(collected)
        tmp = np.nanmedian(tmp, axis=0) if operation=='median' else np.nanmean(tmp, axis=0)
        list_of_median_embs.append(tmp)
    
    assert(len(list_of_dicts) == 0)

    embs_df = pd.DataFrame(list_of_median_embs)
    embs_df.columns = ['emb' + str(i) for i in range(embs_df.shape[1])]

    plate_dfr_smry = pd.concat([plate_dfr_smry, embs_df], axis=1)

    return plate_dfr_smry


def postprocess_embeddings(profiles_df, var_thresh=1e-5, 
                           trt_column='perturbation_id',
                           norm_method='spherize',
                           l2_norm=False):
    emb_features = [c for c in profiles_df.columns if c.startswith('emb')]
    emb_meta_features = [c for c in profiles_df.columns if c not in emb_features]
    # l2-normalize features
    if l2_norm:
        profiles_df[emb_features] = normalize(profiles_df[emb_features], axis=0, norm='l2')
    if norm_method=="no_post_proc":
        return profiles_df
    emb_df = profiles_df.loc[:, emb_features]
    meta_df = profiles_df.loc[:, emb_meta_features]
    
    # filter low-variance features prior to normalization
    emb_vars_samples = np.var(emb_df[meta_df[trt_column] != 'DMSO'], axis=0)
    emb_vars_dmso = np.var(emb_df[meta_df[trt_column] == 'DMSO'], axis=0)
    my_cond = np.logical_and(emb_vars_samples > var_thresh, emb_vars_dmso > var_thresh)

    emb_df_filt = emb_df.loc[:, my_cond]
    profiles_df = pd.concat([meta_df, emb_df_filt], axis=1).reset_index()
    emb_features = [c for c in profiles_df.columns if c.startswith('emb')]
    # per plate normalization methods
    if norm_method in ["standardize", "mad_robustize", "robustize"]:
        norm_prof = []
        samples = (trt_column + " == 'DMSO'") if norm_method == "standardize" else ("~" + trt_column + ".isnull()")
        for plate in profiles_df['plate'].unique():
            norm_df = profiles_df.query('plate == @plate').reset_index(drop=True).copy()
            norm_df = pycytominer.normalize(profiles=norm_df, 
                                                features=emb_features, 
                                                meta_features=emb_meta_features, 
                                                samples=samples, 
                                                method=norm_method, 
                                                output_file='none')
            norm_prof.append(norm_df)
        # concatenate all the plate profiles
        norm_prof = pd.concat(norm_prof).reset_index(drop=True)
        return norm_prof
    # sphering is fit on all negative controls
    elif norm_method == "spherize":
        profiles_df = pycytominer.normalize(profiles=profiles_df, 
                                        features=emb_features, 
                                        meta_features=emb_meta_features, 
                                        samples=(trt_column + " == 'DMSO'"), 
                                        method=norm_method, 
                                        output_file='none')
    else:
        raise ValueError("Invalid norm_method specified")
    return profiles_df

def post_proc(embedding_df, val_df,
              trt_column='perturbation_id',
              cols = ['perturbation_id', 'target'],
                    operation='mean', 
                    norm_method='spherize',
                    l2_norm=False):
    embeddings_proc_well = postprocess_embeddings(embedding_df, 
                                                  var_thresh=1e-5,
                                                  trt_column=trt_column,
                                                  norm_method=norm_method,
                                                  l2_norm=l2_norm)

    emb_features = [c for c in embeddings_proc_well.columns if c.startswith('emb')]
    # aggregates 
    embeddings_proc_agg = pycytominer.aggregate(embeddings_proc_well, 
                                               strata=[trt_column], 
                                               features=emb_features,
                                               operation=operation)
    embeddings_proc_agg = pd.merge(left=embeddings_proc_agg, right=val_df.loc[:, cols].drop_duplicates(), how='left')
    # remove DMSO
    #embeddings_proc_agg = embeddings_proc_agg[embeddings_proc_agg['perturbation_id'] != "DMSO"]
    return embeddings_proc_well, embeddings_proc_agg