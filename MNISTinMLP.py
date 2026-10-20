#用 NumPy 在 MNIST 上训练 MLP
from collections import defaultdict
import math

import dnnlpy
import dnnlpy.models.mlp as mlp
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets #torchvision库提供MNIST数据集

rng = np.random.default_rng(42)
print('Numpy version:', np.__version__)

#准备MNIST数据集
root=dnnlpy.get_data_root()
    #dnnlpy.get_data_root()表示调用dnnlpy库中的get_data_root()函数
    #返回dnnlpy库默认存放在此电脑的所有数据集的根目录路径，赋值给root
#加载MNIST训练集
train_ds=datasets.MNIST(root,train=True,download=True)
    #在torchvision中的dataset中加载MNIST训练集到路径root中
    #返回值是MNIST训练集的访问接口，赋值给train_ds

#加载MNIST测试集
test_ds=datasets.MNIST(root,train=False,download=True)

#数据预处理：定义了训练和测试数据本身，转为NumPy数组、展平、归一化
X_train=train_ds.data.numpy()
    #从MNIST训练集train_ds中取出原始图像张量x，.data存放在data属性中
    #并将其从PyTorch张量转换为NumPy数组，赋值给X_train
X_train = np.reshape(X_train, (-1, 28 * 28))
    #展平
    #np.reshape改变X_train数组形状
X_train = X_train.astype(np.float32) / 255.0
    #归一化：将像素值放缩到[0,1]，让损失值计算平滑稳定
    #astype将X_train的数组元素类型从uint8改为float32
y_train = train_ds.targets.numpy()
    #从MNIST训练数据train_ds中取出标签y，.targets存放在targets属性中
    #并将其从PyTorch张量转换为NumPy数组，赋值给y_train
X_test = test_ds.data.numpy()
X_test = np.reshape(X_test, (-1, 28 * 28))
X_test = X_test.astype(np.float32) / 255.0
y_test = test_ds.targets.numpy()

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

#查看数据图片及其对应label
fig = plt.figure(1, figsize=(8, 2))
axes = fig.subplots(1, 6)
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title(f'label: {y_train[i]}', fontsize=10)
dnnlpy.set_matplotlib_format('highdpi')
plt.show()

#mini-batch迭代器：从传入的X和y中生成mini-batch大小的训练&labels数据，并且打乱顺序
class DataLoader:
    def __init__(
        self,
        X:np.ndarray,
        y:np.ndarray,
        batch_size:int,
        shuffle:bool=True #打乱样本索引顺序
    ):
        self.X=X
        self.y=y
        self.batch_size=batch_size
        self.shuffle=shuffle

    def __iter__(self):
        indices=np.arange(len(self.X))
            #len(self.X)：获取数据集样本总数：60000
            #np.arange(60000)：生成一个从0到59999的整数索引数组indices[0,1...59999]，代表所有样本的索引
        if self.shuffle:
            #如果self.shuffle为True则允许打乱索引数组indices
            rng.shuffle(indices)
                #就用随机数生成器rng把indices数组原地打乱顺序
        #取出一个mini—batch，通过yield把数据给外部调用者
        for start in range (0,len(self.X),self.batch_size):
            idx=indices[start:start+self.batch_size]
            yield self.X[idx],self.y[idx]

    #放回一个epoch中有几个mini-batch
    def __len__(self):
        return math.ceil(len(self.X)/self.batch_size)

#Accuracy:计算准确率
def accuracy(logits:np.ndarray,labels:np.ndarray)->float:
    pred=np.argmax(logits,axis=1) #得到预测出来的类别
        #沿着第1维axis=1在logits中找到获得分数最大的那个类别
    correct=np.sum(pred==labels) #统计预测出来的类别和答案labels相同的个数
    return correct/len(labels)

#定义训练&测试数据迭代器dl、实例化mlp、定义损失函数loss_fn、定义优化器为SGD
train_dl=DataLoader(X_train,y_train,batch_size=128,shuffle=True)
test_dl=DataLoader(X_test,y_test,batch_size=128,shuffle=False)
model=mlp.MLP(input_dim=784,hidden_dim=256,num_classes=10)
loss_fn=mlp.CrossEntropyLoss()
optimizer=mlp.SGD(model.parameters(),lr=0.1)

history=defaultdict(list) #头文件提供
    #记录每个epoch中训练数据：acc、loss等

#训练
num_epochs=10
for epoch in range(1,num_epochs+1): #range左闭右开，训练轮次从1到10
    train_loss=0.0
    correct_samples=0

    #每次从train_dl中取一个mini-batch去训练
    for X,y in train_dl:
        #forward->loss->backward->update
        logits=model(X)
        loss=loss_fn(logits,y)
        dlogits=loss_fn.backward()
        dx=model.backward(dlogits)
        optimizer.step()
        optimizer.zero_grad()

        #每个epoch的损失值累加，统计预测正确的个数
        train_loss+=loss.item()
            #.item()将numpy标量变为python的float类型
        correct_samples+=np.sum(np.argmax(logits,axis=1)==y)

    avg_train_loss=train_loss/len(train_dl) #一个epoch平均每个batch的的平均损失
    avg_train_acc=correct_samples/len(train_ds) #一个 epoch 的训练准确率
    #记录每个epoch的训练数据
    history['loss'].append(avg_train_loss)
    history['acc'].append(avg_train_acc)

    #模型测试
    test_loss=0.0
    correct_samples=0
    for X,y in test_dl:
        logits=model(X)
        loss=loss_fn(logits,y)
        test_loss+=loss.item()
        correct_samples+=np.sum(np.argmax(logits,axis=1)==y)
    avg_test_loss=test_loss/len(test_dl)
    avg_test_acc=correct_samples/len(test_ds)
    history['test_loss'].append(avg_test_loss)
    history['test_acc'].append(avg_test_acc)

    #打印每个epoch训练结果
    n=len(str(num_epochs))
    print(
        f'Epoch[{epoch:{n}d}/{num_epochs:{n}d}]'
        f'|loss:{avg_train_loss:.4f}'
        f'|acc:{avg_train_acc:.4f}'
        f'|test_loss:{avg_test_loss:.4f}'
        f'|test_acc:{avg_test_acc:.4f}'
    )

#每个epoch的train集和test集对应的loss值图像
fig = plt.figure(2, figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
xticks = np.arange(1, num_epochs + 1)
ax.plot(xticks, history['loss'], marker='o')
ax.plot(xticks, history['test_loss'], marker='o')
ax.grid(linestyle='--', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(['train_loss', 'test_loss'])
ax.set_title('MLP Training Loss')
plt.show()
#每个epoch的train集和test集对应的acc值图像
fig = plt.figure(2, figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
xticks = np.arange(1, num_epochs + 1)
ax.plot(xticks, history['acc'], marker='o')
ax.plot(xticks, history['test_acc'], marker='o')
ax.grid(linestyle='--', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(['train_acc', 'test_acc'])
ax.set_title('MLP Training Accuracy')
plt.show()

#模型训练好后，随机不重复地从test中选10张样本去测试
indices=rng.choice(len(X_test),size=10,replace=False)
x_samples=X_test[indices]
y_samples=y_test[indices]
logits=model(x_samples)
y_preds=np.argmax(logits,axis=1) #对sample的预测结果y_preds
#将预测结果y_pred和真实答案y_samples放一起可视化比对
fig = plt.figure(3, figsize=(7, 3.2))
axes = fig.subplots(2, 5)
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_samples[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Pred: {y_preds[i]}, Label: {y_samples[i]}', fontsize=10)
fig.tight_layout()
plt.show()