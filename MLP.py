#Numpy搭建MLP模型
from typing import override

import dnnlpy.models.mlp as mlp
import numpy as np

rng=np.random.default_rng(42)
print('Numpy version:', np.__version__)

#创建MLP模型
class MLP(mlp.Module):
    #定义网络结构：有哪些层、参数形状
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        self.fc1=mlp.Linear(input_dim, hidden_dim) #第一个线性层，输入维数input_dim,输出维数hidden_dim
        self.relu=mlp.ReLU()
        self.fc2=mlp.Linear(hidden_dim, num_classes)

    @override #override表这里定义的forward重写覆盖父类mlp.Module里的forward
    def forward(self,x:np.ndarray)->np.ndarray:
        #x: np.ndarray：参数 x 的类型注解，表示输入的x的是 NumPy 数组
        #-> np.ndarray：返回类型注解，表示该方法返回一个 NumPy 数组
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

    @override
    def backward(self, grad:np.ndarray)->np.ndarray:
        # grad:np.ndarray：参数grad的类型注解，表示输入的grad的是NumPy数组
        # -> np.ndarray：返回类型注解，表示该方法返回一个NumPy数组
        grad=self.fc2.backward(grad)
            # 首次输入的grad是 L 对Z=logits的梯度，作为上游梯度。
            # fc2.backward(grad) 内部做两件事：
            #   1. 用链式法则计算 L 对 W2 和 b2 的梯度，存入fc2.weight.grad 和 fc2.bias.grad。
            #   2. 用上游梯度乘以 logits 对 A 的雅可比矩阵（即 W2^T），得到 L 对 A 的梯度，
            #      返回并赋值给 grad，供上一层继续反向传播。
        grad=self.relu.backward(grad)
            #输入的grad是上一层的L对A的梯度，作为上游梯度
            #relu.backward(grad)执行：用上游梯乘以ReLU的导数掩码举证，得到L对H的梯度，
            #赋值给grad供上一层继续反向传播
            #ReLU 层没有可训练参数，所以它不像fc2.backward那样需要保存参数梯度，只负责把梯度继续往前传
        grad=self.fc1.backward(grad)
            #输入的grad是上一层L对H的梯度，作为上游梯度
            #fc1.backward(grad)执行：
            #   1.上游梯度乘上H对X的雅可比矩阵，得到L对X的梯度（即 W1^T），赋值给grad作为最终梯度输出
            #   2.链式法则计算L对W1和b1的梯度，存入到fc1.weight.grad 和 fc1.bias.grad
        return grad

#实例
batch_size=5
input_dim=4
hidden_dim=8
num_classes=3

x=rng.random((batch_size,input_dim)) #输入值x，二维数组5*4
y=np.array([0,1,2,1,0]) #标签值y

model=MLP(input_dim, hidden_dim, num_classes) #实例化MLP，赋值给model
loss_fn=mlp.CrossEntropyLoss() #定义损失函数loss_fn为交叉熵函数，且内部结合了softmax

optimizer=mlp.SGD(model.parameters(), lr=0.1) #MLP选择SGD作为优化器

#1.forward
logits=model(x)
loss_before=loss_fn(logits,y)

print('Logits shape:', logits.shape)

#2.backward
dlogits=loss_fn.backward()
    #反向传播的起点
    #loss_fn.backward()执行返回L对logits的梯度，赋值给dlogits
dx=model.backward(dlogits)
    #dlogits作为上游梯度进入反向传播fc2 → relu → fc1，最终得到L对X的梯度，赋值给dx
    #dx是L对输入x的梯度，不参与参数更新

#3.update参数
optimizer.step()
    #遍历优化器管理的所有参数：fc1.weight、fc1.bias、fc2.weight、fc2.bias
    # 用backward中保存的每个参数的.grad去更新该参数本身

#参数更新前后的loss对比
logits=model(x) #再进行一次forward，计算出新的loss值
loss_after=loss_fn(logits,y)
print('Loss before update:', loss_before)
print('Loss after update:', loss_after)

#参数和梯度的形状一一对应
for p in model.parameters():
    print(f'Parameter shape:{p.shape}\t|Gradient shape:{p.grad.shape}')
