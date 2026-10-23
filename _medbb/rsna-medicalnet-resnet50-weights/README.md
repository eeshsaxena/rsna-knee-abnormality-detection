---
license: mit
datasets: 
  - MRBrainS18
language: 
  - en
metrics: 
  - 
tags: 
  - MedicalNet
  - medical images
  - medical
  - 3D
  - Med3D
thumbnail: "https://github.com/Tencent/MedicalNet/blob/master/images/logo.png?raw=true"

---
# MedicalNet
This repository contains a Pytorch implementation of [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625). 
Many studies have shown that the performance on deep learning is significantly affected by volume of training data. The MedicalNet project aggregated the dataset with diverse modalities, target organs, and pathologies to to build relatively large datasets. Based on this dataset, a series of 3D-ResNet pre-trained models and corresponding transfer-learning training code are provided.

### License
MedicalNet is released under the MIT License (refer to the LICENSE file for detailso).

### Citing MedicalNet
If you use this code or pre-trained models, please cite the following:
```
    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }
```
### Update(2019/07/30)
We uploaded 4 pre-trained models based on more datasets (23 datasets).
```
Model name             : parameters settings
resnet_10_23dataset.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18_23dataset.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34_23dataset.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50_23dataset.pth: --model resnet --model_depth 50 --resnet_shortcut B
```

Hugging Face repository contribution by:
[Rafael Zimmer](https://www.github.com/rzimmerdev)