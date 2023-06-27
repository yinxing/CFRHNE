batch_size = 32    #批处理大小
lambda_gen = 1e-5  #生成器正则化
lambda_dis = 1e-5  #判别器正则化
n_sample = 16  #样本大小
lr_gen = 0.0001#1e-3  学习率
lr_dis = 0.0001#1e-4
n_epoch = 40 #总共训练次数


sig = 1    #生成器高斯分布的方差
time = 2 # 第几次
walk_num = 6
walk_len = 8
label_smooth = 0.  #平滑

change = 10 #每5次交换高阶低阶采样
d_epoch = 15#每次判别器迭代次数
g_epoch = 7 #每次生成器迭代次数
alaph = 2
inf = 40

n_emb = 128  #嵌入大小

dataset = 'aminer' #使用的数据集
prefix = "data"
file = "dimension"
image_name = dataset + "_dimension_128"
node_size_set = {"yelp" : 3913, "dblp" : 37791, "acm" : 11246, "aminer" : 51012}
type_size_set = {"yelp" : 5, "dblp" : 4, "acm" : 3, "aminer" : 4}
relation_size_set = {"yelp" : 8, "dblp" : 6, "acm" : 4, "aminer" : 6}

node_size = node_size_set[dataset]
t_size = type_size_set[dataset] #节点类型的数目
r_size = relation_size_set[dataset] #关系的数目

#graph_filename = '../data/' + dataset + '/' + dataset + '_triple.dat'
graph_filename = '../data/' + dataset + '_triple.dat'


