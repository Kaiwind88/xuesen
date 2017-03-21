---
layout: post
title:  "三月读论文"
date:   2017-03-18
desc: "read papers"
keywords: "reading, paper, DL"
categories: [Deeplearning]
tags: [papers]
icon: icon-html
---

## **Prototypical Networks for Few-shot Learning**
- idea: there exists an embedding in which points cluster around a **single prototype** for each class.
- solver: 与Matching Networks one-shot learning 相似

## **Deep Sketch Hashing: Fast Free-hand Sketch-Based Image Retrieval**
- sketchs --> natural images {不过现在连素描生成图像GAN都有了呀pix2pix}存在以下问题
	1. 大规模检索
	2. geometric distortion {cross-view}

## **What Your Images Reveal: Exploiting Visual Contents for Point-of-Interest Recommendation**
- user,location,images

<div style="text-align: center">
<img src="/xuesen/static/img/paper20173/paper3-3.PNG"/>
</div>
<br>

## **DiscoGAN**

## **Unsupervised Deep Embedding for Clustering Analysis**
- [code](https://github.com/piiswrong/dec)+[review](http://icml.cc/2016/reviews/231.txt)
- 使用DAE无监督学习 得到初始化的网络权值$\theta$以及聚类中心$\{\mu_j\}_{j=1}^k$
- 使用t-分布 计算嵌入相似度{as soft assignment}==>Q
- 使用KL(P||Q)来优化分布{惯用思路}
- P目标分布的选取:一个常用方案是使用delta分布(1/0)表示，但是it is more natural and flexible to use softer probabilistic targets(review: smart choice)。因此他自己设计了一个P{为什么？没看懂}.
- **竟然还优化聚类中心，对聚类中心求梯度**
- 迭代优化也有贡献	