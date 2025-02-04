# %%
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
from source.inference_utils import forward_inference, aggregate_embeddings_plate, post_proc
from source import SimCLR, MergedChannelsDataset

from source.mae.load_mae import load_pretrained_mae
from source.dino.utils import load_pretrained_dino
import source.dino.vision_transformer as vits

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Self-supervised model', default='simclr',
                    choices=['simclr', 'dino', 'mae'], type=str)
parser.add_argument('--arch', default='vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train, used only for MAE')
parser.add_argument('--ckpt', help='Model checkpoint file', type=str)
parser.add_argument('--valset', help='Validation csv file path',
                            default="data/JUMP_valset.csv", type=str)
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading')
parser.add_argument('--operation', help='How to aggregate crops, FOV, and perturbations', default='mean',  type=str)
parser.add_argument('--norm_method', help='How to norm features', default='all', \
                            choices=["standardize", "mad_robustize",
                                     "spherize", "spherize_mad_robustize", 
                                     "mad_robustize_spherize", "spherize_standardize",
                                     "standardize_spherize",
                                     "no_post_proc", "all"], type=str)
parser.add_argument('-o', '--outdir', help='Output directory')
parser.add_argument("--gpus", nargs="*", type=int,  default=[0, 1])
parser.add_argument('--size', nargs='?', const=224, type=int, default=224)
parser.add_argument('--stride', nargs='?', const=None, type=int, default=None)
parser.add_argument('--l2norm', default=False, action='store_true')
args = parser.parse_args()
print(f"Norm method {args.norm_method}")
if args.norm_method=='all':
    norm_method = ["standardize", "mad_robustize",
                   "spherize", "spherize_mad_robustize", 
                   "mad_robustize_spherize", "spherize_standardize",
                   "standardize_spherize", "no_post_proc",]
else:
    norm_method = [args.norm_method]
# crop size and stride
crop_size = args.size
stride = args.stride if args.stride is not None else crop_size

# output directories
ourdirs = []
if args.norm_method=="all":
    for m in norm_method:
        outdir = os.path.join(args.outdir, m)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        ourdirs.append(outdir)
else:
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ourdirs.append(outdir)
os.system(f"chmod -R 777 {args.outdir}")
# %%
# load the validation set
val_df = pd.read_csv(args.valset)
all_plates = list(val_df['plate'].drop_duplicates())

# %%
cols_to_keep = ['batch','plate', 'well',
                'perturbation_id', 'target']
# channel stats
means = torch.tensor([0.12528631, 0.17596765, 0.14736995, 0.13445823, 0.08349566])
sds = torch.tensor([0.12594905, 0.15605405, 0.16031352, 0.15751939, 0.15773378])

# %%
if args.model == 'simclr':
    arch = 'vit_small_patch16_224' if 'small' in args.arch else 'vit_base_patch16_224'
    ssl_model = SimCLR.load_from_checkpoint(args.ckpt, vit=arch)
    embednet = EmbeddingNet(ssl_model)
elif args.model == "mae":
        embednet = load_pretrained_mae(args.arch , args.ckpt)
elif args.model == 'dino':
    patch_size = 16
    checkpoint_key = "teacher"
    arch = "vit_small" if "small" in args.arch else "vit_base"
    arch = arch.split('_')[0] + '_x' + arch.split('_')[1] if 'x' in args.arch else arch
    model = vits.__dict__[arch](
        patch_size=patch_size,
        drop_path_rate=0.1)
    embednet = load_pretrained_dino(model, args.ckpt,
                                 checkpoint_key, model_name=None,
                                 patch_size=patch_size)
else:
    raise ValueError("Invalid --model argument. Should be one of: [simclr, dino, mae]")

device = torch.device('cuda:' + str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(embednet, device_ids=args.gpus) if torch.cuda.is_available() else embednet
model.to(device)
model.eval()

# %%
# run inference for every plate
transf = transforms.Compose([transforms.Normalize(means, sds)])
all_embs = []

for plt in all_plates:
    val_df_sub = val_df[val_df['plate'] == plt].copy()
    val_df_sub = val_df_sub.reset_index(drop=True)

    platedata = MergedChannelsDataset(val_df_sub, transform=transf, inference=True)

    plateloader = DataLoader(platedata, batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=False, 
                                    drop_last=False)
    plate_embs = []
    for crops_with_metadata in tqdm(plateloader):
        with torch.no_grad():
            crop_embs = forward_inference(model, 
                                          crops_with_metadata['crops'],
                                          crops_with_metadata['labels'], device)
            plate_embs.append(crop_embs.cpu().numpy())
    embeddings = aggregate_embeddings_plate(plate_dfr=val_df_sub,
                                            plate_embs=plate_embs,
                                            my_cols = cols_to_keep,
                                          operation=args.operation)
    all_embs.append(embeddings)
    log_and_print(f"Inference for plate {plt} finished successfully.")
# concatenate all plate embeddings
embedding_df = pd.concat(all_embs, ignore_index=True)

# %%
print(f"Postprocessing embeddings method: {norm_method},  aggregation: {args.operation}")
assert isinstance(norm_method, list), f'norm_method {norm_method} is not a list'
for outdir,norm in zip(ourdirs, norm_method):
    # combinations of sphering + other method
    if 'spherize_' in norm:
        well_embs, _ = post_proc(embedding_df, val_df,
                                operation=args.operation,
                                norm_method='spherize',
                                l2_norm=args.l2norm)
        embeddings_proc_well, embeddings_proc_agg = post_proc(well_embs, val_df,
                                                             operation=args.operation,
                                                             norm_method=norm.replace('spherize_', ''))
    elif '_spherize' in norm:
        well_embs, _ = post_proc(embedding_df, val_df,
                                operation=args.operation,
                                norm_method=norm.replace('_spherize', ''),
                                l2_norm=args.l2norm)
        embeddings_proc_well, embeddings_proc_agg = post_proc(well_embs, val_df,
                                                             operation=args.operation,
                                                             norm_method='spherize')
    # otherwise, normalize using a single method
    else:
        embeddings_proc_well, embeddings_proc_agg = post_proc(embedding_df, val_df,
                                                             operation=args.operation,
                                                             norm_method=norm,
                                                             l2_norm=args.l2norm)
    csv_path_well = os.path.join(outdir, f'well_features.csv')
    csv_path_agg = os.path.join(outdir, f'agg_features.csv')
    embeddings_proc_well.to_csv(csv_path_well)
    embeddings_proc_agg.to_csv(csv_path_agg)
print("Wrote well and consensus embeddings")