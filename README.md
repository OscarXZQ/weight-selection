# Initializing Models with Larger Ones

<p align="center">
<img src="./Weight_Selection.png" width=80% height=80% 
class="center">
</p>

We introduce weight selection, a method for initializing smaller models by selecting a subset of weights from a pretrained larger model.
## Installation

Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Weight Selection

Please run `weight_selection.ipynb` under the conda environment created in the previous section and confirm timm's version is 0.6.12. In `weight_selection.ipynb`, we implement weight selection for initializing smaller models with their larger pretrained teacher within the same model family. Before running experiments, please generate the initialization model file with this notebook first. We provide code for initializing ViT-T with a pretrained ImageNet-21K ViT-S.

## Training

We list commands for training on `ViT-T` and `ConvNeXt-F` on CIFAR100 and ImageNet.
- To run baseline (train from random initialization), remove `--finetune` command.


ViT-T from weight selection on CIFAR100
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny  --warmup_epochs 50 --epochs 300 \
--batch_size 64 --lr 2e-3 --update_freq 1 --use_amp true \
--finetune /path/to/weight_selection \
--data_path /path/to/data/ \
--data_set CIFAR100 \
--output_dir /path/to/results/
```

ConvNeXt-F from weight selection on CIFAR100
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_femto  --warmup_epochs 50 --epochs 300 --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 1 --use_amp true \
--finetune /path/to/weight_selection \
--data_path /path/to/data/ \
--data_set CIFAR100 \
--output_dir /path/to/results/
```

ViT-T from weight selection on ImageNet-1K
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 --use_amp true \
--finetune /path/to/weight_selection \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

ConvNeXt-F from weight selection on ImageNet-1K
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 --use_amp true \
--finetune /path/to/weight_selection \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

## Result

CIFAR-100 and ImageNet-1K results of initializing ViT-T and ConvNeXt-F with weight selection from ImageNet-21K pretrained ViT-S and ConvNeXt-T

| setting        | ViT-T | ConvNeXt-F |
|:------------|:-----:|:----------:|
| train from random init (CIFAR-100)     | 72.4  |   81.3     |
| weight selection (CIFAR-100)   | 81.4  | 84.4          |
| train from random init (ImageNet)    | 73.9 | 76.1       |
| weight selection (ImageNet)  | 75.6  | 76.4         |


## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [Dropout](https://github.com/facebookresearch/dropout) codebase.