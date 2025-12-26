# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dataset_CuPy.mnist import load_mnist
from deep_convnet import DeepConvNet
from common_CuPy.trainer import Trainer
import time

# 设置中文字体支持
try:
    # 尝试使用中文字体
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用英文
    print("Warning: Chinese font not available, using English labels")
    rcParams['font.sans-serif'] = ['DejaVu Sans']

print("=" * 60)
print("开始训练深度卷积神经网络 (GPU加速)")
print("=" * 60)

# 设置CuPy
print("CuPy version:", cp.__version__)
print("CUDA available:", cp.cuda.is_available())
if cp.cuda.is_available():
    print("CUDA device count:", cp.cuda.runtime.getDeviceCount())
    print("CUDA device name:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())

# 记录开始时间
start_time = time.time()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

print(f"训练数据: {x_train.shape[0]} 样本")
print(f"测试数据: {x_test.shape[0]} 样本")
print(f"使用GPU: CuPy")
print("=" * 60)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000,
                  early_stopping_patience=5,  # 早停：5轮没有提升则停止
                  early_stopping_target=0.99)  # 目标准确率：99%
trainer.train()

# 记录结束时间
end_time = time.time()
training_time = end_time - start_time

# 保存参数
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")

print("=" * 60)
print(f"训练总耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
print("=" * 60)

# ========== 可视化训练过程 ==========
print("生成训练过程可视化图表...")

# 创建图表
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Deep Convolutional Neural Network Training Process (GPU Accelerated - CuPy)', 
             fontsize=16, fontweight='bold')

# 1. 训练损失曲线
ax1 = plt.subplot(2, 3, 1)
iterations = np.arange(len(trainer.train_loss_list))
ax1.plot(iterations, trainer.train_loss_list, 'b-', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Iterations', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(trainer.train_loss_list))

# 2. 训练和测试准确率对比
ax2 = plt.subplot(2, 3, 2)
epochs = np.arange(1, len(trainer.train_acc_list) + 1)
ax2.plot(epochs, trainer.train_acc_list, 'o-', label='Train Accuracy', 
         linewidth=2, markersize=6, color='#2E86AB')
ax2.plot(epochs, trainer.test_acc_list, 's-', label='Test Accuracy', 
         linewidth=2, markersize=6, color='#A23B72')
# 标记早停点
if trainer.early_stopped:
    ax2.axvline(x=trainer.stopped_epoch, color='red', linestyle='--', 
                linewidth=2, label=f'Early Stop (Epoch {trainer.stopped_epoch})')
ax2.set_xlabel('Epochs', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Train vs Test Accuracy', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, len(trainer.train_acc_list))
ax2.set_ylim(0, 1.05)

# 3. 准确率提升趋势
ax3 = plt.subplot(2, 3, 3)
if len(trainer.test_acc_list) > 1:
    acc_improvement = np.diff(trainer.test_acc_list)
    ax3.bar(epochs[1:], acc_improvement, color=['green' if x > 0 else 'red' for x in acc_improvement], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Epochs', fontsize=11)
    ax3.set_ylabel('Accuracy Change', fontsize=11)
    ax3.set_title('Test Accuracy Improvement Trend', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

# 4. 训练信息统计
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')

early_stop_info = ""
if trainer.early_stopped:
    early_stop_info = f"""
Early Stopping:
- Stopped at Epoch: {trainer.stopped_epoch}/{trainer.epochs}
- Best Test Accuracy: {trainer.best_test_acc:.4f} ({trainer.best_test_acc*100:.2f}%)
- Patience: {trainer.early_stopping_patience} epochs
- Target Accuracy: {trainer.early_stopping_target:.2%} {'(Reached!)' if trainer.test_acc_list[-1] >= trainer.early_stopping_target else ''}
"""
else:
    early_stop_info = f"""
Training Completed:
- All {trainer.epochs} epochs finished
- No early stopping triggered
"""

info_text = f"""
Training Configuration:
{'='*40}
Network: DeepConvNet
Optimizer: Adam (lr=0.001)
Batch Size: 100
Max Epochs: 20
Train Samples: {x_train.shape[0]:,}
Test Samples: {x_test.shape[0]:,}

Training Results:
{'='*40}
Final Train Acc: {trainer.train_acc_list[-1]:.4f} ({trainer.train_acc_list[-1]*100:.2f}%)
Final Test Acc: {trainer.test_acc_list[-1]:.4f} ({trainer.test_acc_list[-1]*100:.2f}%)
Best Test Acc: {max(trainer.test_acc_list):.4f} ({max(trainer.test_acc_list)*100:.2f}%)
Final Train Loss: {trainer.train_loss_list[-1]:.6f}
Total Time: {training_time:.2f} sec
Avg Time/Epoch: {training_time/len(trainer.train_acc_list):.2f} sec
Acceleration: GPU (CuPy)
{early_stop_info}
"""
ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 5. 损失值分布（平滑曲线）
ax5 = plt.subplot(2, 3, 5)
if len(trainer.train_loss_list) > 50:
    # 使用移动平均平滑曲线
    window_size = 50
    smoothed_loss = np.convolve(trainer.train_loss_list, 
                                np.ones(window_size)/window_size, mode='valid')
    ax5.plot(iterations[:len(smoothed_loss)], smoothed_loss, 
             'r-', linewidth=2, label='Smoothed Loss')
    ax5.plot(iterations, trainer.train_loss_list, 
             'b-', linewidth=0.5, alpha=0.3, label='Raw Loss')
    ax5.legend(loc='upper right', fontsize=9)
else:
    ax5.plot(iterations, trainer.train_loss_list, 'b-', linewidth=1.5)
ax5.set_xlabel('Iterations', fontsize=11)
ax5.set_ylabel('Loss', fontsize=11)
ax5.set_title('Training Loss (Smoothed)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. 准确率对比柱状图
ax6 = plt.subplot(2, 3, 6)
categories = ['Initial\nTrain Acc', 'Final\nTrain Acc', 'Initial\nTest Acc', 'Final\nTest Acc']
values = [trainer.train_acc_list[0], trainer.train_acc_list[-1],
          trainer.test_acc_list[0], trainer.test_acc_list[-1]]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax6.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_ylabel('Accuracy', fontsize=11)
ax6.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
ax6.set_ylim(0, 1.05)
ax6.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}\n({value*100:.2f}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# 保存图表
output_filename = 'training_results_gpu.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"训练过程可视化图表已保存: {output_filename}")

plt.show()

print("=" * 60)
print("训练完成！")
print("=" * 60)
