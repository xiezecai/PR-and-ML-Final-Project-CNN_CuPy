# CNN_CuPy - GPU加速的深度卷积神经网络（PR & ML 大作业）

这是参考鱼书（《深度学习入门：基于Python的理论与实现》）第八章的GPU加速版本，使用CuPy库实现。

## 主要改动

1. **NumPy → CuPy**: 所有NumPy操作替换为CuPy，利用GPU加速计算
2. **早停机制**: 添加基于测试集的早停功能，避免过拟合
   - 目标准确率早停：达到99%自动停止
   - 耐心值早停：连续5轮无提升则停止
3.  **新增训练过程和结果的多方面可视化**
4. **新建文件夹**:
   - `common_CuPy/`: GPU版本的通用函数和层
   - `dataset_CuPy/`: GPU版本的数据集加载
   - `CNN_CuPy/`: GPU版本的主要代码

## 文件说明

### CNN_CuPy/
- `deep_convnet.py`: 深度卷积神经网络模型（GPU版本）
- `train_deepnet.py`: 训练深度网络
- `misclassified_mnist.py`: 显示误分类的MNIST图像
- `half_float_network.py`: 测试半精度浮点数的影响

### common_CuPy/
- `functions.py`: 激活函数、损失函数等（GPU版本）
- `layers.py`: 各种神经网络层（GPU版本）
- `optimizer.py`: 优化器（GPU版本）
- `trainer.py`: 训练器（GPU版本）
- `util.py`: 工具函数，包括im2col/col2im（GPU版本）

### dataset_CuPy/
- `mnist.py`: MNIST数据集加载，自动转换为CuPy数组

## 使用方法

### 前置要求
```bash
pip install cupy-cuda11x  # 在鱼书环境要求下，根据你的CUDA版本选择 
# 或者
pip install cupy-cuda12x
```

### 训练网络
```bash
cd CNN_CuPy
python train_deepnet.py
```

### 查看误分类样本
```bash
python misclassified_mnist.py
```

### 测试半精度浮点数
```bash
python half_float_network.py
```

## 性能优势

使用GPU加速后，训练速度可以提升数倍到数十倍，具体取决于： 1.GPU型号 2.批次大小 3.网络复杂度

## 早停机制

训练器支持两种早停策略：

1. **目标准确率早停** (`early_stopping_target=0.99`)
   - 当测试准确率达到99%时自动停止训练
   - 节省训练时间，避免不必要的计算

2. **耐心值早停** (`early_stopping_patience=5`)
   - 当连续5轮测试准确率没有提升时停止训练
   - 防止过拟合，保持最佳泛化性能

可以在`train_deepnet.py`中调整这些参数：
```python
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000,
                  early_stopping_patience=5,    # 耐心值
                  early_stopping_target=0.99)   # 目标准确率
```


## 网络结构

```
conv - relu - conv - relu - pool -
conv - relu - conv - relu - pool -
conv - relu - conv - relu - pool -
affine - relu - dropout - affine - dropout - softmax
```
识别率可达99%以上。
