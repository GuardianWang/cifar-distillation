## distillation from small networks

Target: large networks that originally overfit the training set
can perform better than the small network used for distillation.

Experiments:
- [x] large network w/o distillation
- [x] large network w/ aggressive augmentation
- [ ] large network w/ distillation
- [ ] large network w/ distillation and aggressive augmentation

|      setting       | augment | distill | test top-1 acc | test top-5 acc | train top-1 acc | train top-5 acc |
|:------------------:|:-------:|:-------:|:--------------:|:--------------:|:---------------:|:---------------:|
| resnet_original_20 |         |         |     67.36      |     90.96      |      88.29      |      98.77      |
|      resnet18      |         |         |     57.04      |     80.42      |      99.97      |     100.00      |
|      resnet18      |    x    |         |     55.01      |     77.69      |      99.97      |     100.00      |
|      resnet18      |         |    x    |                |                |                 |                 |
|      resnet18      |    x    |    x    |                |                |                 |                 |
