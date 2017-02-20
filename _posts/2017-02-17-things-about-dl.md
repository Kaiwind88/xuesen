---
layout: post
title:  "Introduction to Deep Learning"
date:   2017-02-18
desc: "DL简单介绍"
keywords: "DL"
categories: [Deeplearning]
tags: [DL]
icon: icon-html
---
Deep Learning 是一个现在人工智能领域大家都在常常提及的词汇，它进入我们的视野并成为主流是因为最近的几个研究成果吸引了所有的注意力。

谷歌大脑计划可以学习在视频中寻找猫，Facebook能从图片中辨认出头像，百度可以识别图像中的对象以及形状，百度和微软用深度学习做语音识别。深度学习可以让那些拥有多个处理层的计算模型来学习具有多层次抽象的数据的表示。这些方法在许多方面都带来了显著的改善，包括最先进的语音识别、视觉对象识别、对象检测和许多其它领域，例如药物发现和基因组学等。除了这些略高端应用，一些最有声望的深度学习研究者包括像Jeff Dean (Google), Yann LeCun(Facebook), Andrew Ng(Baidu)。

深度学习能够发现大数据中的复杂结构。它是利用BP算法来完成这个发现过程的。BP算法能够指导机器如何从前一层获取误差而改变本层的内部参数，这些内部参数可以用于计算表示。深度学习做的最有兴趣的过程是现在它可能以动态的方式学习怎样抽取出可辨别的特征。然而，传统的机器学习需要许多人力资源去手工的提取特征而机器学习是一个重要的方式去学习权重来权衡这些特征。动态的发现有识别力的特征确实一个巨大的进步。机器现在可以学习到什么是重要的以及什么不是重要的，而以前人必须去提取哪些是潜在的重要的特征，而且冒着忽略重要特征信息的风险去让机器学习方法学习权重。简单的说，深度学习的好处之一是将用非监督式或半监督式的特征学习和分层特征提取的高效算法来替代手工获取特征。现在我们已经有了可训练的特征提取器以及可训练的学习算法。

深度学习用许多层叠的非线性转换单元，这样可以执行特征提取以及特征转换。但这还需要我们对于不同的问题手动的搭建不同的学习架构。所以，下一步就应该是怎么根据问题自行的组织层次架构,前一段时间Arvix上有一篇论文就介绍了如何自适应的学习神经网络的深度[Learning The Architecture of Deep Neural Networks]。很典型的模型那个是深度学习用许多ANN的循环层构建了许多更为复杂的生成模型想Deep Belief Networks和Deep Bolzmann Machines.深度卷积网络[CNN]()在处理图像、视频、语音和音频方面带来了突破，而递归网络**RNN**在处理序列数据，比如文本和语音方面表现出了闪亮的一面。下图就是一个简单的神经网络的例子，其中包括输入层，隐藏层，以及输出层。
<!-- ![ANN](https://raw.githubusercontent.com/icodingc/notes/master/pictures/ann.png) -->
<div style="text-align: center">
<img src="https://raw.githubusercontent.com/icodingc/notes/master/pictures/ann.png"/>
</div>
<br>

一个基础的假设是每层单元都会从前边的层中学习抽象的概念。这在图片中可以更直观的感受，第一层可能学习基础的特征例如颜色，边角，简单纹理，第二层学习一些人脸的部分组成，第三层学习更为复杂的特征也就是不同的人脸。因此，学习系统就可以学习到一些高维度的特征，而用统计学规律是发现不了的。学习不是由单个神经元完成的而是由许多层激活神经元构建的神经网络完成的。
<div  style="text-align:center">    
<img src="https://raw.githubusercontent.com/icodingc/notes/master/pictures/faces.png"/>
</div>

就像上图，随着网络的深度越深，学习到的特征就会更加复杂。这样我们就可以用它来做图像识别或者其他相关的任务，下图就是用深度学习坐的图像分类的任务，可以分辨出Faces,Cars,Chairs etc.。

<div  style="text-align:center"> 
<img src="https://raw.githubusercontent.com/icodingc/notes/master/pictures/multiple.png"/>
</div>
<br>

随着网络结构更加的复杂，计算需求就更加的迫切，所以高级的硬件对于基于计算的深度学习来说是很重要的。尤其，强有力的图形处理器(GPUs)就非常适合机器学习中的矩阵和向量运算，GPUs可以加速学习算法从运行时间几周到几小时。这就允许我们创造更加复杂的模型以及更深的网络从而学习更加好的特征代表。下边一张图片也可以帮我们理解不同的层的神经网络学习到的不同特征：第一层学到了颜色，边角等信息，第二层学到了简单纹理特征，第三层学习到了更加复杂的特征，反正就是层数越多学习到的特征就会越复杂。
<div  style="text-align:center">  
<img src="https://raw.githubusercontent.com/icodingc/notes/master/pictures/layers.png"/>
</div>
神经网络的训练一般都是基于反向传播算法，权重更新是一般是通过Stochastic Gradient Descent （SGD），公式如下
<div  style="text-align:center">  
<img src="https://raw.githubusercontent.com/icodingc/notes/master/pictures/gradient.png"/>
</div>
当然还有其他优化算法像Adagrad,RMSprop,Adam etc。下图是几个优化器优化效率的比较（下图本是gif如显示不好请到原来地址）[原gif地址](http://cs231n.github.io/assets/nn3/opt1.gif)，一般情况下最有效率的的Rmsprop & SGD + Momentum ，不过具体问题还的具体分析，优化器的选取还是依据问题而定。
<div  style="text-align:center"> 
<img src="http://cs231n.github.io/assets/nn3/opt1.gif" width = "350" height = "250"/>
</div>
<br>
<!-- ![optim](http://cs231n.github.io/assets/nn3/opt1.gif) -->
这样两层之间的权重就会基于损失函数的梯度更新，eta是学习速率，Google建立了一个异步分布式随即梯度下降服务器，为了完成从视频中识别出动物或者其他对象任务它用了16000GPUs独立的更新权重。其实关于神经网络的一个优化本身就是一个大的问题，现在已经有好多优化技巧，比如，Regularization正规化能有效的防止模型overfitting,Dropout也是正规化的一种方式；神经网络的权值默认方式也是有技巧性的更好的权值默认方式能让神经网络更快的收敛到最优解；然后是在输入或者梯度中加入退火梯度噪音也是一种优化方式能增加模型的鲁棒性；另外不同的非线性映射函数也会产生不同的效果，现在来说Relu（Rectified Linear Unit）是最常用的激活函数了，它能有效的的消减梯度消失或者梯度爆炸情况，不仅如此，不同的损失函数也会产生不同的效果，一般现在常用的是hinge loss 和 cross-entorpy loss 了，相对应的就是最后一层神经元用了SVM 或者 是softmax分类器，神经网络优化本身就是一门学问。

CNN
-----------
在深度学习的图像相关处理方面的一个非常强有力的方法是卷积神经网络Convolutional  networks(用了卷积，权值共享等)下图就是一个CNN的例子Lenet，它包括两个卷积层，连个max-pooling 层，在当时的图像分类中达到了state-of-the-art。
![CNN](https://raw.githubusercontent.com/icodingc/notes/master/pictures/lenet5.png) <br>
其实CNN不仅可以用在图像这种特征相对好处理的任务中，在最近的几篇相关NLP任务论文中就有人用不共享权值，不同卷积的CNN做特征提取然后做POS,chunking等nlp分类任务，我们其实就可以把CNN当成一个好的特征提取器，最近有一篇论文解释和可视化了CNN提取特征的过程[Visualizing and Understanding Convolutional Networks]()能帮助我们更好的理解CNN的特征提取功能。

下边是深度学习的一个应用，它用了24个layers来完成annotating images的任务达到了6.6%的错误率，这个结果已经能和人类识别想媲美了。
![lasts](https://raw.githubusercontent.com/icodingc/notes/master/pictures/last.png) 

Embedding 是根据深度学习引入的另外的一个概念。Embedding 是为了避免数据稀疏的问题。例如，我们可以从文档中抽取出单词，并且根据选择的上下文窗口创造出words embedding.Word embedding 就是一个参数化的函数吧一些语言映射到一个高维度空间(大概是200-500)。训练的embedding vectors对于语言模型任务来说具有很有意思的特性，它可以表示一些线性的概念对于相似的词汇，例如capitals 和 countries，the queen 和 the king。下表描述了一些关系对：<br>
![Word2vec](https://raw.githubusercontent.com/icodingc/notes/master/pictures/word2vec.png) 
![table](https://raw.githubusercontent.com/icodingc/notes/master/pictures/relations.png) 

RNN
---------
除了以上介绍的，还有一种叫做循环神经网络（Recurrent Networks）。首次引入反向传播算法时，最令人兴奋的便是使用RNNs神经网络训练。对于涉及到序列输入的任务，比如语音和语言，利用RNNs能获得更好的效果。RNNs一次处理一个输入序列元素，同时维护网络中隐式单元中隐式的包含过去时刻序列元素的历史信息的“状态向量”。如果是深度多层网络不同神经元的输出，我们就会考虑这种在不同离散时间步长的隐式单元的输出，这将会使我们更加清晰怎么利用反向传播来训练RNNs，下图是一个RNN架构的例子，尽管RNN可以宣称说能解决long-term 依存问题，但是效果还不是那么的明显，所以就有人研究更为复杂的RNN例如现在很火的LSTM以及GRU等。这些复杂的结构不仅减小了RNN训练的难度而且能捕捉更前面的信息。[Visualizing and Understanding Convolutional Networks]()是一篇理解RNN的论文。<br>
![RNN](https://raw.githubusercontent.com/icodingc/notes/master/pictures/rnn.jpg) 

除了CNN RNN NTM这些复杂的架构，其实深度学习中还后很多好好的idel例如Attention 机制等就能很好的优化RNN不能很好解决long-term依存的问题，因为它能捕捉前边所有上下问的信息，下边是Attention的一个例子。

<div  style="text-align:center"> 
<img src="http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.16.08-PM-235x300.png"/>
</div>

上图中用BRNN来生成h向量，而y代表翻译过程，正规化向量a={a1,a2,a3...}就代表每个预测分别连接h1,h2...ht的权重，而这个向量a也是可以解释的，下图中横坐标是对应的翻译的英文单词，纵作标是对应的德语单词，而中间的像素值越大就代表a的权值越大，也就越表明两个单词越有关系。

<div  style="text-align:center">    
<img src="http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.23.48-PM-279x300.png"/>
</div>

后记：本文其实是为了应付研究生某课程。因本人水平以及篇幅有限不能把深度学习神经网络许多东西概括下来。如果兴趣了解更深的知识请学习相关的课程或者书籍。