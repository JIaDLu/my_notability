## self-supervised learning 

> supervised learning 
>
> 有训练集有标签

![image-20241022154800873](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410221548029.png)

在这个过程中并没有用到label的资料，因此，self-supervised可以看成是一种unsupervised learning的一种方法（有label就是supervised learning，没有label就是unsupervised learning ）

### 搞出BERT

BERT的架构就是transformer encoder。它的输入一般是：一排文字。还可以是：

* 文字  --- > vector
* 图片  --- > vector
* 音频  --- > vector

#### masking input

这里的token就是你在处理文本的单位。这个token的大小是你自己决定的。 在中文里面，我们通常把方块字当成一个token。

输入一些文字，这段文字中的一些方块字会被随机地盖住 

* 把句子中的某一个字换成一个特殊的符号，也就是一个special token （可以想成是一个新的中文的字，但这个字在你的字典中从来没有出现过的）
* 随机把这个句子中的这个字换成任意的一个方块字，随便找一个中文字把它替换掉。

![image-20241022161352041](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410221613152.png)

:radio:BERT如何知道要猜什么？

模型在预测被遮盖的词时，确实需要有明确的“标签”来指导训练，这也是MLM的关键步骤之一。

> 自监督学习的本质就是在**不依赖人工标注**的情况下，从数据本身自动生成“标签”来进行训练。换句话说，虽然我们没有手动标注的数据，但我们可以利用输入数据的一部分来充当“标签”，从而进行模型的训练。
>
> 在BERT的Masked Language Modeling（MLM）中，模型自己**生成了训练所需的“标签”**，这个标签就是被遮盖（mask）掉的词本身。

在训练BERT时，**模型是有监督地学习**的，并且每个被遮盖（masked）词是有对应的正确答案（标签）的。

* **掩码后，保留了真实词汇**：虽然模型看到的是带有[MASK]的输入句子，但训练数据中的每一个被遮盖的词都有它对应的原始词作为标签。换句话说，模型输入是遮蔽后的句子，但是它知道每个被遮蔽位置的正确词是什么。

* **损失函数（Loss function）引导模型学习**：在训练过程中，模型会根据它的预测输出和真实的被遮蔽词进行比较。使用**交叉熵损失**（cross-entropy loss），模型会计算它预测的词和真实词之间的差距。损失函数会引导模型不断调整参数，让它尽可能接近正确的预测结果。

  举个例子： 输入句子：“I [MASK] natural language processing.”

  * 真实的标签（目标词）：**“love”**。
  * 模型可能会输出几个猜测，比如：love、like、enjoy等。但最终，模型会根据上下文选择最接近“love”的词作为它的输出。

#### Next sentence prediction

收集很多的句子

有两句话，中间表示SEP句子的分割，在句子最前面有一个特别的符号CLS

![image-20241022161722792](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410221617837.png)

:sailboat:我们只关注CLS的输出，这个输出作linear network，再做一个二分类，得到yes/ no

这个yes / no 就是要预测说 这两个句子是不是相接的，如果是相接的，就输出yes，如果不是相接的，就输出no。

### 让BERT学了这两个任务，那怎么用呢？

学两个任务 --- > 

* 怎么做填空题
*  预测两个句子是不是应该被接在一起（有研究表明好像是没有什么用的）

因此：最主要的就是 教会了bert如何做填空题，难道它就只会做填空题吗？不是

这个巨大的bert可以被看成是一个pre-train的model，它已经学到了人类语言中的一些通用的东西。

:card_file_box:downstream tasks

这些下游任务可能跟填空题不一定要有关系，可能甚至都没有关系，但是bert可以被用在这些任务上。这些任务就是bert真正被使用的任务就是叫做 <u>下游任务</u>。

> 当我们应用到下游任务的时候，我们还是需要一些标注的。

![image-20241022171311239](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410221713367.png)

##### Case 1

让bert来进一步解决sentiment analysis的问题

![image-20241022171639036](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202410221716113.png)

















