## MOE 专家模型



### 基础概念

`conditional computation`：这个概念被提出是为了提高模型的效率和推理速度。

:sailboat:简单来说，条件计算指的是在每次执行模型时，不是激活整个模型的所有参数和层，而是根据输入数据的特征，有选择性地激活一部分模型的层数和层。这种方式可以避免不必要的计算，从而减少计算成本，提高模型的效率。

#### 举个例子

假设我们有一个非常大的神经网络，通常我们每次输入数据时，所有的神经元（层）都会被激活并进行计算。但在**条件计算**中，模型会根据输入数据的复杂度或某些特定的条件来**动态决定哪些部分需要被激活**，而哪些可以跳过。例如：

* 对于简单的输入，模型可能只需要用较浅的层来得到结果；
* 对于复杂的输入，模型才会激活更多的层或参数进行更复杂的计算。

#### 应用场景

1. **大模型加速**：大语言模型（like GPT）参数非常多，如果对简单任务也启用所有参数，成本非常高。条件计算可以让模型在简单任务时“省力”。
2. **低资源设备**：在资源受限的设备（如手机或嵌入式设备）上，条件计算可以减少对计算资源的需求，从而实现快速推理。
3. **任务自适应**：当模型面对不同任务时，它可以根据任务的难度或特征来选择不同的计算路径。

`the entire model is activated`: 激活整个模型的代价

:sailboat:在典型的深度学习模型中，当我们进行训练时，**每个输入样本都会激活整个模型的所有参数和层**。这意味着每次输入时，所有神经元都参与计算，这在计算量上是非常庞大的。

* **模型规模**：随着模型参数数量的增加（例如，层数增加或每层神经元数量增加），每次前向传播（forward pass）和反向传播（backward pass）需要的计算量都会显著增加。
* **训练样本数量**：如果训练数据的数量也增加，比如从1万增加到10万，模型必须在每个训练迭代中对每个样本进行完全计算。

### One of Conditional Computation: Sparsely-Gated Mixture-of-Experts

由多达数千个前馈子神经网络组成的稀疏门控 混合专家层

一个可训练的门控(gating)网络可确定每个示例使用这些专家的稀疏组合
$$
\begin{bmatrix}
0\\
0\\
0\\
1\\
0\\
0\\
0\\
0
\end{bmatrix}
$$
一系列expert networks，共有n个

有一个gating network，叫做G，它的输出是一个n维的稀疏向量

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc(x)
    
class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)
    
    def forward(self, x):
        gate_score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        return output

input_size = 5
output_size = 3
num_experts = 4
batch_size = 10

model = MoELayer(num_experts, input_size, output_size)

demo = torch.randn(batch_size, input_size)

output = model(demo)

print(output.shape)  # 输出: torch.Size([10, 3])
```

