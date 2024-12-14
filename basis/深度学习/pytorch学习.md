## modelDescription

## Fundation

### torch基础:

1. `torch.nn`==(神经网络模块)==

   提供构建神经网络的基本构件，包括各种层，激活函数，损失函数等

   * `nn.Linear`:用于构建全连接层。

   * `nn.Conv2d`: 用于构建卷积层

     卷积层：使用了 3x3 的卷积核在输入图像上进行卷积操作

     ```python
     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
     ```

     :large_orange_diamond:在卷积操作中，in_channels 表示输入数据的通道数，也就是输入图像的"深度"。如果处理的是一个 RGB 彩色图像，通常会有 3 个通道（红、绿、蓝）

     **<u>一个</u>**卷积核的大小是 3x3x3，最后的3表示卷积核的深度必须和输入通道数一致。卷积核的深度必须与输入的通道数一致。这样，卷积核在“扫描”输入图像时，会同时处理所有通道的信息，得到一个特征图。

     :national_park:Out_channels=16表明：<u>**有 16 个独立的卷积核**</u>，每个卷积核都会与输入图像进行卷积操作。

     使用多个卷积核的目的是从输入数据中提取不同的特征。每个卷积核会通过训练学习到不同的模式或特征，例如边缘、色彩、形状等。因此，模型在这一层生成 16 个不同的特征图，从而捕捉输入图像的更多特征信息。

     你可以把每个卷积核想象成一个“特征探测器”，16 个卷积核可以同时探测不同的特征，这为后续层提供更丰富的特征表示。

   * `nn.ReLu`: 激活函数，引入非线性特征，提高模型表达能力

     :sailboat:定义：在神经网络的每一层中，数据经过输入、线性变换(如矩阵乘法和加偏置)后，进入激活函数。这些函数会对输入数据进行非线性变换，输出结果给下一层。

     :palm_tree:为什么需要激活函数：假设没有激活函数，神经网络的每一层都只是线性组合。即便你堆叠再多的层，也只能表现出线性变化的关系，就好像只是简单的加权求和。因此，这样的网络无法处理复杂的问题。它会“太简单”，解决不了现在中复杂的任务。

     激活函数则引入了“非线性”，就像给模型增加了一种更复杂的表达能力，使得它可以“弯曲”数据，拟合更复杂的关系，学习更加深层的特征。

2. `torch.optim`==（优化器模块）==

   * `optim.SGD`：随机梯度下降优化器，适合需要控制学习率和动量的模型。

     :cactus:复习梯度下降

     **向量**是具有大小（长度）和方向的量。我们通常用一个箭头来表示，箭头的长度代表向量的大小，箭头的方向表示它的方向。例子：假设你在一张平面图上，从点 A(0,0) 移动到点 B(3,4)。从 A 到 B 的位移可以用向量 v⃗=(3, 4) 来描述。这里的 3 表示水平方向上的移动，4 表示垂直方向上的移动。

     **梯度（Gradient）** 是一个多维空间中的向量，它描述了一个函数在某个点的变化趋势和变化速率。

     对于一个标量函数 f(x)（例如，损失函数），其中 x=[x1,x2,…,xn]是输入向量，梯度定义为对每个变量的偏导数所组成的向量：

     ![image-20241117125049008](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411171251409.png)

     ![image-20241117130546570](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411171305598.png)

     1. 简单高效：由于只需计算一个样本或小批量数据的梯度，速度较快。
     2. 可能收敛到局部最优：当面对复杂或非凸的损失函数时，容易陷入局部最优解或收敛缓慢。
     3. 需要手动调整学习率：学习率的选择对训练效果影响较大。学习率过大可能导致发散，过小则收敛速度变慢。

     SGD 支持 **动量（Momentum）**，通过加入动量项来减少震荡现象，从而加快收敛速度。

   * `optim.adam`: 目标是在每次参数更新时，不仅利用当前的梯度信息，还考虑梯度的历史信息和变化幅度，从而实现参数的智能化调整。

     > [!NOTE]
     >
     > 动量：动量表示物体在运动中保持当前状态的“惯性”。质量越大或速度越快，动量越大，物体更难改变方向或停止。

     想象一个小球滚下山坡。如果只依赖当前坡度决定滚动方向（即普通梯度下降），小球可能因为地形不平或“震荡”左右摇摆，走得很慢。

     引入**动量**后，小球会“记住”之前的运动方向（惯性），即使遇到小坡度的地方，也能沿着之前积累的方向前进，减少震荡并加速下坡。

     :walking_woman:梯度下降中的振荡现象与动量作用

     为了让参数更新更加平稳，优化过程可以“记住”之前的方向，像一个有惯性的物体一样，减少震荡。这就是动量的作用。

     ![image-20241120222057076](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411202220227.png)

     vt: 表示当前的“速度”，是历史梯度的累积。

     β: 动量因子（类似摩擦系数，通常取 0.9），用于控制历史信息的衰减。

     就像物体在一个方向上移动时，速度 vt 不仅受当前的推力（梯度）影响，还会保留之前的运动趋势。

3. `torch.utils.data`==（数据加载模块）==

   简化和优化数据加载过程。包括 `Dataset` 和 `DataLoader`，可以方便地实现数据集的加载、处理和批次化。

   :gift:当你处理一个大型数据集时，通过 `DataLoader` 进行**批次化**和**随机打乱数据顺序**可以显著提高训练效率和模型的泛化能力。

4. `torch.Tensor`==（张量操作模块）==

   张量是 PyTorch 的基本数据结构，与 Numpy 的数组类似，但更适用于 GPU 加速计算。

   * `torch.tensor(data)`：将数据转换为张量。
   * `.to(device)`：将张量转移到特定设备（如 GPU）
   * 各种操作（如 `.reshape()`, `.view()`, `.sum()`, `.mean()` 等）用于数据变换和基本运算。

### Data

##### Tensor结构

Scalar, Vector, Matrix, 多维张量(3维及以上)

* **多维性（Multidimensionality）**：Tensor可以扩展到任意维度，例如4维用于视频（时间+高+宽+通道）。
* **数据类型（Data Type）**：常见数据类型包括float32, int64, bool等，适应不同计算需求。
* **设备兼容性（Device Compatibility）**：Tensor可以驻留在CPU或GPU上，通过框架如PyTorch轻松切换。

:japanese_ogre:构建案例

```python
import torch
# 1. 从列表创建
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor([[1, 2],
        [3, 4]])
# 2. 随机初始化
tensor2 = torch.rand((3, 3))
tensor([[0.7656, 0.8728, 0.7888],
        [0.3349, 0.4173, 0.2478],
        [0.6170, 0.5612, 0.8763]])
# 3. 特定值初始化
zeros = torch.zeros((2, 2))  # 全0张量
ones = torch.ones((2, 2))    # 全1张量
tensor([[0., 0.],
        [0., 0.]])
tensor([[1., 1.],
        [1., 1.]])
```

:musical_keyboard:操作方式

```python
import torch
# 矩阵加法
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = a + b  # 加法
d = torch.matmul(a, b)  # 矩阵乘法
tensor([[ 6,  8],
        [10, 12]])
tensor([[19, 22],
        [43, 50]])
# 维度变换
reshaped = a.view(-1)  # 展平成一维
print(reshaped)
tensor([1, 2, 3, 4])
```

广播机制（Broadcasting）是深度学习框架中一种简化操作的技巧，用于处理**不同形状的张量（Tensor）之间的运算**。

当两个张量的形状不完全相同时，但在某些维度上满足特定规则，框架会通过**隐式扩展**较小的张量，使其在逻辑上具有相同的形状，从而允许它们进行逐元素运算。

规则：

1. **从右往左对齐维度**：逐一比较两个张量的维度。
2. **维度匹配**：
   * 如果某一维度相同，直接匹配。
   * 如果某一维度为1，可以扩展为另一张量的大小。
   * 如果维度不同且都不为1，则不兼容，报错。

```python
import torch
# 模拟批量数据和偏置
data = torch.rand((4, 3))  # 4个样本，每个样本3个特征
bias = torch.tensor([1, 2, 3])  # 偏置为 1, 2, 3
# 广播加法
result = data + bias  # 自动将 bias 广播为 (4, 3)
print(result)
```

##### Normalization or standardization

归一化（或标准化）是针对每个特征（列）单独进行的操作

```python
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
```

MinMaxScaler 的归一化原理：MinMaxScaler`会对输入的每一个特征（列）分别计算最小值和最大值：

![image-20241209205806172](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202412092058906.png)

##### DataSplit

:japan:**​`train_test_split`**

train_test_split是 scikit-learn`中的一个函数，主要用于将数据集划分为训练集和测试集（或验证集）。它的作用是确保模型训练和评估使用的数据不重叠，从而避免数据泄漏。

参数与要点：

1. arrays：输入数据，可以是特征矩阵和标签，也可以是多个数组。
2. test_size: 指定测试集占总数据的比例。例如，test_size=0.1`表示测试集占10%。
3. random_state: 设置随机种子以保证分割的结果可复现。
4. shuffle: 是否在分割数据之前打乱数据。

```python
from sklearn.model_selection import train_test_split
# 示例数据
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 0, 1]
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
训练集: [[7, 8], [1, 2], [5, 6]]
测试集: [[3, 4]]
训练标签: [1, 0, 0]
测试标签: [1]
```

:hamster:`torch.utils.data.DataLoader`

| 功能模块         | 作用                                                         |
| ---------------- | ------------------------------------------------------------ |
| train_test_split | 将数据集划分为训练集和测试集，确保模型训练和评估互不干扰。   |
| DataLoader       | 用于批量加载数据，支持动态数据加载（分批加载，减少内存占用）、随机打乱（shuffle）以及数据增强等功能。 |

train_test_split 的典型作用

* **数据划分**：通常用于早期预处理阶段，比如 `sklearn` 的函数可以快速完成这一步。
* **比例设置**：控制训练集与测试集的比例，比如 80% 用于训练，20% 用于评估。

DataLoader的典型作用

`torch.utils.data.DataLoader` 是 PyTorch 数据加载的核心工具，用于将数据集打包为可迭代对象，支持高效的批量加载、数据打乱、并行处理等功能。

* **高效加载**：用于训练和验证阶段，按批（batch）取数据，提升 GPU/CPU 的利用率。
* **预处理支持**：结合 `torchvision.transforms`，进行数据增强（如图像旋转、裁剪等）。
* **分布式支持**：便于多线程处理，优化大规模数据集的加载。

:old_key:示例一：加载内置数据集

```python
'''
在torchvision.datasets中，
我们用compose来封装一系列需要执行的操作，可以做图像变换、数据增强，totensor,以及归一化等操作。
'''
train_loader = torch.utils.data.DataLoader(
    							torchvision.datasets.MNIST('./data/', train=True, download=True,   
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                               (0.1307,), (0.3081,))])),
                  batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,   
                  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                               (0.1307,), (0.3081,))])),
                  batch_size=batch_size_test, shuffle=True)   
```

:unicorn:示例二： 自定义数据集与加载

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 定义自定义数据集
class CSVDataset(Dataset):  # 这个类继承自 torch.utils.data.Dataset，用于定义一个自定义数据集
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)  # 加载 CSV 文件

    def __len__(self):
        return len(self.data)  # 数据长度

    def __getitem__(self, idx): # 用于规定对数据特征的提取方式
        row = self.data.iloc[idx]  # 获取第 idx 行数据
        features = torch.tensor(row[:-1].values, dtype=torch.float32)  # 特征
        label = torch.tensor(row[-1], dtype=torch.long)  # 标签
        return features, label

# 加载数据
dataset = CSVDataset('data.csv')  # 假设 CSV 文件中最后一列是标签
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 遍历数据加载器 每次循环从 loader 中取出一批数据（特征 + 标签），直到取完所有数据。
for batch_idx, (features, labels) in enumerate(loader): 
    print(f"Batch {batch_idx + 1}:")
    print(f" - Features: {features.shape}")
    print(f" - Labels: {labels.shape}")
    break

```

重点解释`__getitem__` 的作用

是 Dataset 的核心方法，用来定义如何根据索引 idx 返回单个样本。它的作用和用法如下

1. 接收索引：通过 idx 定位数据集中的某一行。
2. 提取数据：通过 self.data.iloc[idx] 取出第 idx 行的数据。
3. 分离特征和标签：
   * row[:-1] 提取除最后一列外的所有列，作为 特征。
   * row[-1] 提取最后一列，作为 标签。
4. 转换为 Tensor：
   * 特征转换为 float32 类型的 Tensor，适合神经网络输入。
   * 标签转换为 long 类型的 Tensor，适合分类任务中的交叉熵损失计算。

> 用DataLoader将 CSVDataset 对象包装成支持批量加载的可迭代对象。
>
> 参数解析：
>
> * `batch_size=16`：一次epoch内的训练中，每次丢进去训练就是 16 个样本，直至全部丢完，这个epoch就结束。
> * `shuffle=True`：在每个 epoch 开始时随机打乱数据顺序。

### Training

##### Hypothesis Function 假设函数

假设函数是模型的核心，用于根据输入x预测输出´ŷ (即预测值)

* 假设函数是将输入数据映射到输出预测值的一个数学模型。
* 它直接影响预测值 ŷ，从而决定损失函数的计算。

在深度学习中，假设函数是**神经网络的前向传播部分**，通常由你定义的模型（继承自 torch.nn.Module）决定。

:ice_cream:默认情况下不需要单独定义假设函数

通常，你定义的神经网络模型本身就已经实现了假设函数。模型的==forward()==方法负责处理输入数据并生成预测值。

```python
import torch
import torch.nn as nn
# 定义一个简单神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 3)  # 输入 10 维特征，输出 3 类
		
    # 模型 forward 方法隐式地定义了假设函数：
    def forward(self, x):
        return self.fc(x)  # 假设函数隐式定义：y = Wx + b

# 模型实例
model = SimpleModel()
# 输入数据
x = torch.randn(5, 10)  # 5 个样本，每个样本 10 维
ŷ = model(x)  # 假设函数通过 forward 方法计算预测值
print(f"Predictions: {ŷ}")
```

:taxi:假设函数与损失函数的关系

在训练过程中：

* 假设函数生成预测值 `ŷ`。
* 损失函数通过比较预测值 `ŷ` 和真实值 `y`，计算误差并指导模型更新参数。

```python
# 假设有预测值和真实值
ŷ = torch.tensor([0.2, 0.7, 0.1])  # 假设函数的输出
y = torch.tensor([0, 1, 0])  # 真实值（分类任务）

# 定义损失函数
loss_fn = nn.BCELoss()  # 二元交叉熵损失
loss = loss_fn(ŷ, y.float())
print(f"Loss: {loss.item()}")

```

> 假设函数控制模型的预测值。
>
> 损失函数控制预测值和真实值的差距。

对于复杂任务，比如对比学习或生成模型，假设函数需要显式地自定义。例如：

* **对比学习**：假设函数可以定义为两个特征向量的余弦相似度。
* **生成对抗网络（GAN）**：生成器和判别器都需要各自的假设函数。

:collision:总结：

==不需要显式定义的情况：==

* 使用标准的神经网络模型（如 `nn.Linear`、卷积层等）时，假设函数已经隐式融入到 `forward` 方法中。
* 你只需专注于设计网络结构和训练流程。

==需要显式定义的情况：==

* 使用传统机器学习算法（如线性回归、逻辑回归）线性回归假设函数：y = Wx + b。逻辑回归假设函数是 Sigmoid函数的应用
* 自定义复杂的模型行为，比如特殊的映射函数或多目标优化。

##### Loss function  损失函数

损失函数是一个衡量模型预测值（output）与真实值（ground truth）之间差异的数学函数。其输出的标量值（标量：scalar）反映模型当前性能，帮助优化器（optimizer）指导模型参数更新。通过计算损失值，优化器（如 SGD、Adam）会指导模型参数如何更新。

> [!IMPORTANT]
>
> 这个scalar在实际操作中是会被计算出来的。这个值代入优化器函数(梯度下降算法函数)，通过反向传播计算梯度，用于更新模型权重。
>
> 1. **损失值计算**
>
>    * 损失函数接收模型的预测值和真实值，计算两者之间的差异。
>    * 输出的标量值（scalar）反映了当前模型预测与目标的偏离程度。
>
> 2. 传递到优化器
>
>    * 通过反向传播（backpropagation），PyTorch 会根据损失值对每一层的参数计算梯度（gradient）。
>    * 损失值本身不直接参与梯度下降，==而是通过其对模型参数的导数（梯度）影响权重更新==。
>    * 梯度的大小和方向决定了权重调整的幅度和方向。损失值仅是反映当前模型状态的一个中间结果。
>
>    ```markdown
>    损失值是评估指标，而梯度是优化工具。梯度引导权重更新，使模型更接近目标！
>    ```
>
> 3. 梯度下降优化
>
>    * 优化器（如 `SGD`、`Adam`）利用这些梯度，结合预设的学习率（learning rate），逐步调整模型参数，以最小化损失值。
>    * 优化器（如 SGD、Adam）会指导模型参数如何更新

```python
[损失函数输出 scalar] -> [反向传播计算梯度] -> [优化器更新权重] 
```



##### Optimizer 优化器

在训练过程中，优化器的主要任务是：

1. 根据损失函数计算的梯度更新模型参数。
2. 利用不同策略（如动量、学习率调整）提高训练效率，避免陷入局部最优。

##### Procedure 训练流程

模型训练的核心步骤包括：

* 前向传播计算损失；
* 反向传播计算梯度；
* 优化器更新模型参数。

### Model

##### Convolutional Neural Networks: 

1. nn.Conv2d：表示卷积层
2. nn.MaxPool2d：表示最大池化层
3. nn.Linear： 表示全连接层

:satellite:==Simple CNN Example==

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义第一个卷积层，输入通道为3（RGB图像），输出16个特征图，使用3x3卷积核
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 定义第二个卷积层，将16个特征图转换为32个特征图
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 定义一个池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义一个全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 10)  # 输入特征维度为 32 * 8 * 8，输出为10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一个卷积层 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二个卷积层 + 激活函数 + 池化
        x = x.view(-1, 32 * 8 * 8)           # 将特征图展平成一维向量
        x = self.fc1(x)                      # 全连接层
        return x
```

正确的任一个卷积池化层的操作：**“卷积 -> 激活 -> 池化”**

:hamburger:为什么：

* 卷积操作本身是线性的，如果不使用激活函数，那么无论堆叠多少卷积层，模型的表达能力依然会受到限制。**非线性特征的捕获**：卷积操作后，激活函数能够立即对特征图进行非线性映射，使得网络能够识别复杂模式和结构，提升模型的表达能力。将激活函数与卷积层紧密配合使用，可以更好地捕捉数据中的细节和模式。

  <u>不使用：</u>“卷积 -> 池化 -> 激活”的顺序

* **池化操作的作用**：池化层主要是降维和特征选取（如最大池化保留局部区域内的最大值）。如果在池化操作后才进行激活函数的应用，那么激活函数的作用就被弱化了。池化已经在一定程度上简化了特征图，激活函数的非线性作用变得不那么显著。

* **减少信息丢失**：池化会降低特征图的空间分辨率，因此在池化之前进行激活操作可以确保模型捕捉到尽可能多的非线性特征。否则，池化操作可能会丢失一些重要的特征信息，使得后续的激活无法充分发挥作用。

:vatican_city:需要被学习的参数

> **需要学习的层**：卷积层（核权重与偏置）、全连接层、BatchNorm 等。
>
> **不需要学习的层**：池化层（无论输入数据如何变化，池化层只会根据设定好的池化窗口大小和步幅执行固定的取最大值或平均值等操作）。

##### GRU

**Gated Recurrent Unit (GRU)** 是一种改进的循环神经网络（Recurrent Neural Network, RNN），它通过引入门控机制来更好地捕获序列中的长期依赖信息，同时缓解传统 RNN 的梯度消失问题。

核心机制：

1. 更新门：决定当前时间步的信息是否保留以及历史信息是否需要更新。
2. 重置门：决定当前时间步的信息是否需要结合之前时间步的隐藏状态。

通过这些门控机制，GRU 能够动态调节对序列中不同时间步的关注程度。它比 LSTM (Long Short-Term Memory) 更加简单，参数更少，计算效率更高。

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义 GRU 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)
```

##### Attention

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        embed_size: 输入嵌入向量的维度
        heads: 多头注意力中头的数量
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        # 定义 Query, Key, Value 的线性变换
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
```





