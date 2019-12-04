# nlp-project-01
针对SIF算法中的参数a进行优化
评价标准：对句子进行向量化，计算出sentence_embedding，然后计算两个句子之间的余弦距离作为相似度，将其与实际相似度（来自数据集的人工标注）计算pearsonr值。pearsonr值高的说明embedding效果更好
根据论文中提供的经验，最优参数通常为1e-4或者1e-3，因此在0.0001在0.001之间，以0.0001为步进进行搜索。
代码及结果见ipynb文件，最优参数为1e-4
所用数据集：https://github.com/IAdmireu/ChineseSTS
