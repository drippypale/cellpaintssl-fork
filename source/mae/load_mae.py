import argparse
from pathlib import Path
import torch
import torch.nn as nn

from source.mae.util.pos_embed import interpolate_pos_embed
import source.mae.models_vit as models_vit
from source.dino.utils import _load_checkpoint_compat, trunc_normal_


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--model",
        default="vit_small_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir_mae",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir_mae", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def _normalize_model_name(model_name: str) -> str:
    name = str(model_name).lower()
    # Common repo passes names like 'vit_small_patch16_224'; models_vit uses '*_patch16'
    if name.endswith("_224"):
        name = name[:-4]
    return name


def load_pretrained_mae(
    model_name, ckpt_path, remove_head=True, drop_path=0.1, global_pool=True
):
    model_key = _normalize_model_name(model_name)
    if model_key not in models_vit.__dict__:
        raise KeyError(
            f"Unknown MAE model '{model_name}'. Try one of: {', '.join([k for k in models_vit.__dict__.keys() if k.startswith('vit_')])}"
        )

    model = models_vit.__dict__[model_key](
        drop_path_rate=drop_path,
        global_pool=global_pool,
    )

    checkpoint = _load_checkpoint_compat(ckpt_path, map_location="cpu")
    print("Load pre-trained checkpoint from: %s" % ckpt_path)
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    if global_pool:
        assert set(msg.missing_keys) == {
            "head.weight",
            "head.bias",
            "fc_norm.weight",
            "fc_norm.bias",
        }
    else:
        assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    # vit's classification head for imagenet
    if remove_head:
        model.head = nn.Identity()
        model.fc_norm = nn.Identity()
    else:
        trunc_normal_(model.head.weight, std=2e-5)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Number of params (M): %.2f" % (n_parameters / 1.0e6))
    return model


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = load_pretrained_mae(args.model, args.finetune)

    x = torch.rand(2, 5, 224, 224)
    y = model(x)
    print(y.shape)
