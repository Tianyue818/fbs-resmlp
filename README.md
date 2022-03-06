# FBS-ResMLP on Cifar-10
## Requirements
+ Python 3.7 >
+ PyTorch 1.4 >
+ torchvision 0.5 >
+ numpy, tqdm, math
## Implementation
+ Dataset: CIFAR-10
+ Model: Res-MLP
+ Optimizer: Adam(1e-2)
+ Batch Size: 256
+ ResMLP dim: 384
+ ResMLP depth: 12
+ ResMLP Patch size: 8×8
+ sparsity ratio: 0.5
## Hyperparameter
+ Lr: 1e-2 (weight decay:1e-5)
+ Layer scale: 1e-5
## 
```
    python main.py
```
## References
[1] Gao, Xitong, et al. "Dynamic channel pruning: Feature boosting and suppression." [[arxiv]](https://arxiv.org/abs/810.05331) (ICLR 2019).

[2] Hugo Touvron, et al. "ResMLP: Feedforward networks for image classification with data-efficient training"[[arxiv]](https://arxiv.org/abs/2105.03404) (CVPR 2021).
