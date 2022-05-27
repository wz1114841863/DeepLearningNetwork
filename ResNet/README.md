# ResNet / ResNeXt
```
ResNet / ResNeXt 网络实现
├── ResNet:
    ├── lib: 模型存放文件夹
        ├── __init__
        ├── models
            ├── __init__
            ├── ResNet: 网络实现
    ├── pretrained_pth：预训练权重
    ├── result: 训练结果存储路径
    ├── tools：训练/验证脚本文件
        ├── _init_paths：添加 ./ResNet 到sys.path
        ├── pred_imgs: 存放几张用于预测的花的图片，注意不能是数据集中的图片
        ├── train: 训练脚本
        ├── load_weights: 迁移学习加载预训练权重的几种方法
        ├── predict：用训练后的权重去预测图片种类
        ├── batch_predict：用训练后的权重去批量预测图片种类
```