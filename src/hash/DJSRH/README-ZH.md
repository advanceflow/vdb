# DJSRH
***********************************************************************************************************

这个存储库用于 ["深度联合语义重构哈希用于大规模无监督跨模态检索"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Deep_Joint-Semantics_Reconstructing_Hashing_for_Large-Scale_Unsupervised_Cross-Modal_Retrieval_ICCV_2019_paper.pdf) 

(即将在2019年ICCV, 口头报告中发布)

作者：Shupeng Su\*, [Zhisheng Zhong](https://zzs1994.github.io)\*, Chao Zhang (\* 作者同等贡献)。


***********************************************************************************************************
## 目录
- [简介](#简介)
- [使用方法](#使用方法)
- [消融研究](#消融研究)
- [与SOTAs的比较](#与SOTAs的比较)
***********************************************************************************************************

## 简介

跨模态哈希将多媒体数据编码到一个共同的二进制哈希空间中，其中可以有效地测量不同模态样本之间的相关性。深度跨模态哈希通过深度神经网络生成更多语义相关的特征和哈希码，进一步改善了检索性能。在本文中，我们研究了无监督深度跨模态哈希编码，并提出了深度联合语义重构哈希（DJSRH），它有以下两个主要优势。首先，为了学习保留原始数据邻域结构的二进制码，DJSRH构建了一个新颖的联合语义亲和矩阵，巧妙地整合了不同模态的原始邻域信息，从而能够捕捉输入多模态实例的潜在内在语义亲和力。其次，DJSRH后续训练网络生成最大限度地重建上述联合语义关系的二进制码，通过所提出的重建框架，这对批量训练更有能力，因为它重建特定的相似性值，而不是像常见的拉普拉斯约束仅保留相似性顺序。广泛的实验表明，DJSRH在各种跨模态检索任务中都显著提高了性能。

<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/DJRSH.png" width="90%" height="90%"></div align=center>

***********************************************************************************************************

## 使用方法
### 要求
- python == 2.7.x
- pytorch == 0.3.1
- torchvision
- CV2
- PIL
- h5py

### 数据集
对于数据集，我们遵循[深度跨模态哈希的Github (Jiang, CVPR 2017)](https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab)。您可以从以下链接下载这些数据集：
- 维基百科文章，[链接](http://www.svcl.ucsd.edu/projects/crossmodal/)
- MIRFLICKR25K, [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EpLD8yNN2lhIpBgQ7Kl8LKABzM68icvJJahchO7pYNPV1g?e=IYoeqn)], [[百度网盘](https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew), 密码: 8dub]
- NUS-WIDE (前10概念), [[OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/zszhong_pku_edu_cn/EoPpgpDlPR1OqK-ywrrYiN0By6fdnBvY4YoyaBV

5i5IvFQ?e=kja8Kj)], [[百度网盘](https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ), 密码: ml4y]


### 过程

__以下实验结果为平均值，如果您需要更好的结果，请多运行几次实验（2~5次）。__

- 克隆此仓库：`git clone https://github.com/zzs1994/DJSRH.git`。
- 将`settings.py`中的'DATASET_DIR'更改为您放置数据集的位置。
- 训练模型的示例：
```bash
python train.py
```
- 修改`settings.py`中的参数`EVAL = True`以进行验证。
- 消融研究（__可选__）：如果您想评估我们DJSRH的其他组件，请参考我们的论文和`settings.py`。

***********************************************************************************************************

## 消融研究
表1. 在NUS-WIDE上评估DJSRH中每个组件的有效性的mAP@50结果。

<center>

模型|配置|64位 (I→T)|64位 (T→I)|128位 (I→T)|128位 (T→I)|
|:---------:|:---:|:-----:|:----:|:----:|:----:|
DJSRH-1|S=S<sub>I</sub>|0.717|0.712|0.741|0.735|
DJSRH-2|S=S<sub>T</sub>|0.702|0.606|0.734|0.581|
DJSRH-3|&beta;S<sub>I</sub>+(1−&beta;)S<sub>T</sub>|0.724|0.720|0.747|0.738|
DJSRH-4|+(&eta;=0.4)|0.790|0.745|0.803|0.757|
DJSRH-5|+(&mu;=1.5)|0.793|0.747|0.812|0.768|
DJSRH|+(&lambda;<sub>1</sub>=&lambda;<sub>2</sub>=0.1)|__0.798__|__0.771__|__0.817__|__0.789__|
DJSRH-6|−(&alpha;=1)|0.786|0.770|0.811|0.782|

</center>

从表中我们可以观察到我们提出的每个组件对最终结果都有一定的作用。

***********************************************************************************************************

## 与SOTAs的比较
表2. 在各种编码长度和数据集上的图像查询文本（I→T）和文本查询图像（T→I）检索任务的mAP@50结果。最佳性能显示为<font color="red">红色</font>，次优显示为<font color="blue">蓝色</font>。
<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/results.png" width="90%" height="90%"></div align=center

图1. 在128编码长度的不同数据集上的精确度@top-R曲线。
<div align=center><img src="https://github.com/zzs1994/DJSRH/blob/master/page_image/results_curve.png" width="90%" height="90%"></div align=center

***********************************************************************************************************
## 引用

如果您觉得这个代码有用，请引用我们的论文：

	@inproceedings{su2019deep,
		title={Deep Joint-Semantics Reconstructing Hashing for Large-Scale Unsupervised Cross-Modal Retrieval},
		author={Shupeng Su, Zhisheng Zhong, Chao Zhang},
		booktitle={International Conference on Computer Vision},
		year={2019}
	}

所有权利由作者保留。
***********************************************************************************************************
