# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset_CuPy.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

sampled = 1000 # 为了实现高速化
x_test = x_test[:sampled]
t_test = t_test[:sampled]

print("caluculate accuracy (float64) ... ")
print(float(cp.asnumpy(network.accuracy(x_test, t_test))))

# 转换为float16型
x_test = x_test.astype(cp.float16)
for param in network.params.values():
    param[...] = param.astype(cp.float16)

print("caluculate accuracy (float16) ... ")
print(float(cp.asnumpy(network.accuracy(x_test, t_test))))
