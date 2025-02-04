# Self-supervised learning (SSL) for Cell Painting
This repository contains the code used for training, inference, and evaluation of SSL models presented in the manuscript "Self-supervision advances morphological profiling by unlocking powerful image representations." In this study, we applied self-supervised learning (SSL) methods for feature extraction from Cell Painting images. 


**Note: All of the instructions assume that the working directory is the cloned repo / unzipped code directory**

## Table of Contents
- [Self-supervised learning (SSL) for Cell Painting](#self-supervised-learning-ssl-for-cell-painting)
  - [Table of Contents](#table-of-contents)
  - [1. Conda environment](#1-conda-environment)
    - [1.1 Linux (CUDA)](#11-linux-cuda)
    - [1.2 Linux / Mac OS (CPU, x86\_64/arm64)](#12-linux--mac-os-cpu-x86_64arm64)
  - [2. Download model checkpoints, images and embeddings](#2-download-model-checkpoints-images-and-embeddings)
  - [3. Training SSL models on NVIDIA GPUs](#3-training-ssl-models-on-nvidia-gpus)
    - [3.1 DINO](#31-dino)
    - [3.2 MAE](#32-mae)
    - [3.3 SimCLR](#33-simclr)
  - [4. Feature extraction using SSL models](#4-feature-extraction-using-ssl-models)
  - [5. Evaluation of representations](#5-evaluation-of-representations)
  - [6. Performance comparison of bioactivity prediction models using Cell Painting](#6-performance-comparison-of-bioactivity-prediction-models-using-cell-painting)
  - [7. License](#7-license)


## 1. Conda environment
We provide conda environments for Linux with CUDA and Linux/Mac OS CPU-only (x86_64 and arm64 architectures). **Note that model training can only be conducted in a Linux environment with NVIDIA GPUs.**
### 1.1 Linux (CUDA)
Create a conda environment for training, inference and evaluation of SSL models for Linux with NVIDIA GPUs:
```
conda create -n ssl_cellpaint python=3.8
conda activate ssl_cellpaint
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
**Note**: make sure to adjust the CUDA version (`cu111`) for installing PyTorch. To determine your CUDA version, run `nvidia-smi | grep CUDA`.
### 1.2 Linux / Mac OS (CPU, x86_64/arm64)

Create a conda environment for inference and evaluation of SSL models (CPU version compatible with Linux/Mac OS):
```
conda create -n ssl_cellpaint python=3.9
conda activate ssl_cellpaint
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## 2. Download model checkpoints, images and embeddings
**Note: `wget` is needed to download the data**
To download pretrained model checkpoints, test images and evaluation set embeddings, run the following command:
```
bash download_data.sh
```
This may take a couple of minutes as the data is about 6 GB.

The downloaded and unzipped `SSL_data` folder contains:
+ `checkpoints`: best checkpoints for SSL models trained on single-source data
+ `test_images`: 384 multichannel images from a single JUMP target-annotated plate
+ `dataload`: CSV file with image paths pointing to `test_images/JCPQC051`
+ `embeddings`: precomputed SSL representations of all evaluation sets used in the study, including the single-source, multisource and gene overexpression (ORF) datasets. 


In case the download script doesn't work, please use the following [link](https://www.dropbox.com/sh/ezdq6413dooy24w/AACPXxVydjciEnTqMNuHQyeka?dl=1) to download the test data and place it in the working directory, renaming it as `SSL_data`.
## 3. Training SSL models on NVIDIA GPUs
**Note that training is only supported in a Linux environment with NVIDIA GPUs.**

We trained all SSL models on NVIDIA Tesla-V100 GPU's with 32GB GPU memory using PyTorch's `DistributedDataParallel` multi-GPU paradigm. You might have to adjust the number of GPUs used to avoid "out of memory" issues on your machine.

Here, we provide examples of using the code for training DINO, MAE and SimCLR using the toy dataset in `SSL_data/test_images`. Note that this dataset has only one plate and is intended only for testing the code.
### 3.1 DINO
To train DINO with the ViT-S backbone for 200 epochs, run the following command:
```
export CUDA_VISIBLE_DEVICES=0,1 && python -m torch.distributed.launch --master_port 47781 --nproc_per_node=2 train_dino.py --arch vit_small --lr 0.004 \
 --path_df SSL_data/dataload/dataset.csv \
 --epochs 200 --freeze_last_layer 3 --warmup_teacher_temp_epochs 20 --warmup_teacher_temp 0.01 \
 --norm_last_layer true --use_bn_in_head false  --momentum_teacher 0.9995 \
 --warmup_epochs 20 --use_fp16 false --batch_size_per_gpu 64 --num_workers 6 \
  --output_dir SSL_data/checkpoints/DINO-ViT-S
```

This command launches the training process for DINO using the ViT-S backbone. Here's a breakdown of the different arguments and options used:

+ `export CUDA_VISIBLE_DEVICES=0,1`: Sets the CUDA visible devices to GPU devices 0 and 1.
+ `--arch`: Specifies the architecture of the DINO model (e.g. ViT-S).
+ `--lr`: Sets the base learning rate for training.
+ `--path_df`: Specifies the path to the CSV file containing the training dataset image paths.
+ `--epochs`: Sets the total number of training epochs.
+ `--freeze_last_layer`: Freezes the last layers during training.
+ `--warmup_teacher_temp_epochs 20 --warmup_teacher_temp 0.01`: Applies a warm-up schedule to the teacher temperature, gradually increasing it from 0.01 to the desired value over the specified number of epochs.
+ `--norm_last_layer true`: Enables normalization of the last layer of the model.
+ `--use_bn_in_head false`: Disables the use of batch normalization in the model's head.
+ `--momentum_teacher`: Sets the momentum factor for updating the teacher network's parameters.
+ `--warmup_epochs`: Sets the number of learning rate warm-up epochs.
+ `--use_fp16 false`: Disables mixed precision training using half-precision (float16).
+ `--batch_size_per_gpu`: Sets the batch size per GPU during training.
+ `--num_workers`: Specifies the number of worker processes for data loading.
+ `--output_dir`: Sets the output directory where the checkpoints and training logs will be saved.

### 3.2 MAE
To train MAE with the ViT-S backbone for 200 epochs, run the following command:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 && python -m torch.distributed.launch --master_port 47785 \
 --nproc_per_node=4 train_mae.py --world_size 4 \
 --path_df SSL_data/dataload/dataset.csv \
 --accum_iter 1  --model mae_vit_small  --samples_per_image 4   --batch_size 64 \
 --norm_pix_loss  --mask_ratio 0.5  --epochs 200  --warmup_epochs 30 \
 --blr 1.5e-4  --weight_decay 0.05 --output_dir  SSL_data/checkpoints/MAE-ViT-S
```

This command initiates the training process for MAE using the ViT-S backbone. Here's a breakdown of the command arguments:

+ `export CUDA_VISIBLE_DEVICES=0,1,2,3`: Sets the CUDA visible devices to GPU devices 0, 1, 2, and 3.
+ `--world_size`: Sets the total number of processes for distributed training.
+ `--path_df`: Specifies the path to the CSV file containing the training dataset image paths.
+ `--accum_iter`: Sets the number of gradient accumulation steps.
+ `--model`: Specifies the model architecture (e.g. ViT-S).
+ `--samples_per_image`: Specifies the number of image crops per input image.
+ `--batch_size`: Sets the batch size per GPU for training.
+ `--norm_pix_loss`: Use (per-patch) normalized pixels as targets for computing loss.
+ `--mask_ratio`: Sets the ratio of masked patches during training.
+ `--epochs`: Sets the total number of training epochs.
+ `--warmup_epochs`: Linear rate warmup epochs.
+ `--blr`: Sets the base learning rate for training.
+ `--weight_decay`: Specifies the weight decay factor for regularization.
+ `--output_dir`: Specifies the output directory where checkpoints and training logs will be saved.

### 3.3 SimCLR
To train SimCLR with the ViT-S backbone for 200 epochs, run the following command:
```
python train_simclr.py --arch vit_small_patch16_224 \
 --gpus 0 1 --batch_size 256 --num_workers 6 \
 --path_df SSL_data/dataload/dataset.csv --max_epochs 200 --ckpt_path SSL_data/checkpoints/SimCLR-ViT-S \
 --every_n_epochs 10 --lr 0.001 --wd 0.1
```
This command initiates the training process for SimCLR using the ViT-S backbone. Here's a breakdown of the command arguments:

+ `--arch`: ViT architecture  to be used. It defaults to "vit_small_patch16_224" if not provided.
+ `--gpus`: Specifies the list of GPUs to use for training
+ `--batch_size`: Specifies the batch size for training. Defaults to 256 if not provided.
+ `--num_workers`: Specifies the number of workers used for data loading. Defaults to 6 if not provided.
+ `--max_epochs`: Specifies the maximum number of training epochs.Defaults to 200 if not provided.
+ `--ckpt_path`: Specifies the path to save model checkpoints.
+ `--every_n_epochs`: Specifies the frequency of saving checkpoints. By default, a checkpoint is saved every 10 epochs.
+ `--path_df`: Specifies the path to the data file (CSV format) used for training.
+ `--lr`: Maximum learning rate warmed up over 30 epochs from `min_lr=1e-6`. Defaults to 0.001.
+ `--wd`: Weight decay value. Defaults to 0.1 if not provided.
## 4. Feature extraction using SSL models
Note: for CPU-based inference,  `--num_workers` should be set to 0. First, run the following command to set the number of workers based on GPU availability:
```
NUM_WORKERS=$(if command -v nvcc >/dev/null 2>&1; then echo 6; else echo 0; fi)
```

The following commands will use a model checkpoint saved in `SSL_data/checkpoints` and run inference on test images in `SSL_data/test_images` for DINO, MAE or SimCLR. **The embeddings will be saved in the directory specified by the output flag `-o`.**

To run inference on test images using DINO:
```
python inference.py --model dino --arch vit_small_patch16 \
 --ckpt SSL_data/checkpoints/DINO-ViT-S_singlesource/checkpoint0200.pth \
 --batch_size 32 --num_workers $NUM_WORKERS \
 -o SSL_data/embeddings/testdata/DINO --valset SSL_data/dataload/dataset.csv \
 --size 224 --stride 224 --gpus 0 --norm_method spherize_mad_robustize
```

For MAE:
```
python inference.py --model mae --arch vit_small_patch16 \
 --ckpt SSL_data/checkpoints/MAE-ViT-S_singlesource/checkpoint0200.pth \
 --batch_size 32 --num_workers $NUM_WORKERS \
 -o SSL_data/embeddings/testdata/MAE --valset SSL_data/dataload/dataset.csv \
 --size 224 --stride 224 --gpus 0 --norm_method spherize_mad_robustize
```

For SimCLR:
```
python inference.py --model simclr --arch vit_small_patch16 \
 --ckpt SSL_data/checkpoints/SimCLR-ViT-S_singlesource/checkpoint0200.pth \
 --batch_size 32 --num_workers $NUM_WORKERS \
 -o SSL_data/embeddings/testdata/SimCLR --valset SSL_data/dataload/dataset.csv \
 --size 224 --stride 224 --gpus 0 --norm_method spherize_mad_robustize
```

## 5. Evaluation of representations
The following commands will execute the evaluation code (`evaluate.py`) to compare the performance of precomputed representations in `SSL_data/embeddings`, generating CSV files containing evaluation metrics and producing plotted figures as output. By default, the `--basedir` argument is set to `SSL_data/embeddings`. The `-i` argument is used to specify the subdirectory within the basedir where multiple representation types (e.g. DINO, MAE, CellProfiler) are located and need to be compared.


To evaluate the single-source representations, execute the following command:
```
python evaluate.py -i singlesource
```
The CSV files with evaluation metrics and plotted figures  will be saved in `SSL_data/embeddings/singlesource`.

For evaluating the multisource representations, use the following command:
```
python evaluate.py -i multisource
```
The CSV files and figures for the multisource dataset will be saved in `SSL_data/embeddings/multisource`.

To evaluate the ORF embeddings, execute the following command with the `--orf` flag provided to ensure correct evaluation adapted for ORF perturbations:
```
python evaluate.py -i ORF --orf
```
The CSV files and figures for the gene overexpression dataset will be saved in `SSL_data/embeddings/ORF`.

## 6. Performance comparison of bioactivity prediction models using Cell Painting

To train a fully connected network on top of Dino featuers, execute the following command:

```
python supervised_baseline/hti-cnn-master/main_supervised.py --config supervised_baseline/hti-cnn-master/configs/fnn_dino.json --g 1 --j 2 --training.batchsize 128
```

To train a fully connected network on top of Cellprofiler featuers, execute the following command:

```
python supervised_baseline/hti-cnn-master/main_supervised.py --config supervised_baseline/hti-cnn-master/configs/fnn_cell-profiling.json --g 1 --j 2 --training.batchsize
```

To evaluate the previous models on the testset, execute the following command:

```
python supervised_baseline/hti-cnn-master/eval.py --config supervised_baseline/hti-cnn-master/configs/fnn_cell-profiling_test.json (OR) --config supervised_baseline/hti-cnn-master/configs/fnn_dino_test.json --g 4 --j 2
```

## 7. License
The code is distributed under the BSD 3-Clause License, which grants users the freedom to use, modify, and distribute the code with certain obligations and limitations. Please refer to the LICENSE file for detailed information regarding the terms and conditions.

The model weights provided in `SSL_data/checkpoints` are intended for non-commercial use only. These model weights are licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [license](https://creativecommons.org/licenses/by-nc/4.0/legalcode).