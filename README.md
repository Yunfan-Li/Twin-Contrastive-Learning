# Twin Contrastive Learning for Online Clustering (TCL)

This is the code for the paper "Twin Contrastive Learning for Online Clustering" (IJCV 2022). 

TCL extends the previous work "Contrastive Clustering" (AAAI 2021, https://github.com/Yunfan-Li/Contrastive-Clustering) by selecting most confident predictions to finetune both the instance- and cluster-level contrastive learning.

TCL proposes to mix weak and strong augmentations for both image and text modality. More performance gains are observed by the twin contrastive learning framework compared with the standard instance-level contrastive learning.

The code supports multi-gpu training.

Paper Link: https://link.springer.com/article/10.1007/s11263-022-01639-z

# Environment

- diffdist=0.1
- python=3.9.12
- pytorch=1.11.0
- torchvision=0.12.0
- munkres=1.1.4
- numpy=1.22.3
- opencv-python=4.6.0.66
- scikit-learn=1.0.2
- cudatoolkit=11.3.1

# Usage

TCL is composed of the training and boosting stages. Configurations such as model, dataset, temperature, etc. could be set with argparse. Clustering performance is evaluated during the training or boosting.

## Training

The following command is used for training on CIFAR-10 with a 4-gpu machine,

> OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train.py

## Boosting

The following command is used for boosting on CIFAR-10 with a 4-gpu machine,

> OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 boost.py

## Clustering on ImageNet
To clustering datasets like ImageNet with a large number of classes, a reasonable batch size is needed. However, considering the gpu memory consumption, we recommend inheriting the moco v2 pretrained model (https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar) and freezing part of the network parameters (see details in the manuscript and model.py).

# Dataset

CIFAR-10, CIFAR-100 could be automatically downloaded by Pytorch. For ImageNet-10 and ImageNet-dogs, we provided their indices from ImageNet in the "dataset" folder.

To run TCL on ImageNet and its subsets, you need to prepare the data and pass the image folder path to the `--data_path` argment.

# Citation

If you find TCL useful in your research, please consider citing:
```

```

or the previous conference version
```
@inproceedings{li2021contrastive,
  title={Contrastive clustering},
  author={Li, Yunfan and Hu, Peng and Liu, Zitao and Peng, Dezhong and Zhou, Joey Tianyi and Peng, Xi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={10},
  pages={8547--8555},
  year={2021}
}
```
