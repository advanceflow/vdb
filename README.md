# 单模态数据集
1. CIFAR-10 
   全部都可以先在CIFAR-10上实验。
2. NUS-WIDE
  
# 单模态数据哈希
1. DSH
2. ADSH
3. DJSRH


# 多模态数据集
| 数据集        | 文本形式     | 字段                 | 评价指标       |
|---------------|--------------|----------------------|----------------|
| MIRFLICKR-25K | 多个标签     | 图像＋文本＋主题类别 | mAP            |
| NUS-WIDE      | 多个标签     | 图像＋文本＋主题类别 | mAP            |
| Flickr30K     | 自然语言描述 | 图像＋文本           | R@1, R@5, R@10 |
| MSCOCO        | 自然语言描述 | 图像＋文本           | R@1, R@5, R@10 |
	
	
## 在MIRFLICKR-25K数据集上
1. 训练时利用图像文本对。
2. 测试时，使用文本检索图像，如果图像和文本属于同一主题类别，在评价矩阵中对应位置值为1, 否则为0, 依据评价矩阵计算mAP。


# 多模态数据哈希

