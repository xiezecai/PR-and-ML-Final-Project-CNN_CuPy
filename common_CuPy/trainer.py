# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import cupy as cp
from common_CuPy.optimizer import *

class Trainer:
    """进行神经网络的训练的类
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True,
                 early_stopping_patience=None, early_stopping_target=None):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        
        # 早停参数
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_target = early_stopping_target
        self.best_test_acc = 0.0
        self.patience_counter = 0
        self.early_stopped = False
        self.stopped_epoch = 0

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = cp.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(float(cp.asnumpy(loss)))
        if self.verbose: print("train loss:" + str(float(cp.asnumpy(loss))))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(float(cp.asnumpy(train_acc)))
            self.test_acc_list.append(float(cp.asnumpy(test_acc)))

            if self.verbose: 
                print("=== epoch:" + str(self.current_epoch) + 
                      ", train acc:" + str(float(cp.asnumpy(train_acc))) + 
                      ", test acc:" + str(float(cp.asnumpy(test_acc))) + " ===")
            
            # 早停检查
            current_test_acc = float(cp.asnumpy(test_acc))
            
            # 检查是否达到目标准确率
            if self.early_stopping_target is not None and current_test_acc >= self.early_stopping_target:
                if self.verbose:
                    print(f"*** 达到目标准确率 {self.early_stopping_target:.2%}! 早停训练 ***")
                self.early_stopped = True
                self.stopped_epoch = self.current_epoch
                return True  # 返回True表示应该停止训练
            
            # 检查是否有提升
            if self.early_stopping_patience is not None:
                if current_test_acc > self.best_test_acc:
                    self.best_test_acc = current_test_acc
                    self.patience_counter = 0
                    if self.verbose:
                        print(f"*** 新的最佳测试准确率: {self.best_test_acc:.4f} ***")
                else:
                    self.patience_counter += 1
                    if self.verbose:
                        print(f"*** 无提升计数: {self.patience_counter}/{self.early_stopping_patience} ***")
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        if self.verbose:
                            print(f"*** 早停: {self.early_stopping_patience} 轮无提升 ***")
                        self.early_stopped = True
                        self.stopped_epoch = self.current_epoch
                        return True  # 返回True表示应该停止训练
        
        self.current_iter += 1
        return False  # 返回False表示继续训练

    def train(self):
        for i in range(self.max_iter):
            should_stop = self.train_step()
            if should_stop:
                break

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(float(cp.asnumpy(test_acc))))
            if self.early_stopped:
                print(f"训练在第 {self.stopped_epoch} 轮早停")
                print(f"最佳测试准确率: {self.best_test_acc:.4f}")
            else:
                print(f"完成全部 {self.epochs} 轮训练")
