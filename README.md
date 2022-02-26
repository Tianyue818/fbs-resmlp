# FBS-ResMLP on Cifar-10
## Requirements
+ Python 3.7 >
+ PyTorch 1.4 >
+ torchvision 0.5 >
+ numpy, tqdm, math
## Implementation
+ Dataset: CIFAR-10
+ Model: Res-MLP
+ Lambda: 1e-8
+ Batch Size: 256
+ Optimizer: Adam(1e-4)
## 
```
    python main.py

## References
[1] Gao, Xitong, et al. "Dynamic channel pruning: Feature boosting and suppression." [https://arxiv.org/abs/810.05331] (ICLR 2019).
[2] Hugo Touvron, et al. "ResMLP: Feedforward networks for image classification with data-efficient training"[https://arxiv.org/abs/2105.03404](CVPR 2021).
