batch_size = 32 #批处理大小
n_sample = 16  #样本大小
n_epoch = 10 #总共训练次数

pos_size = 4
neg_size = 32

alaph = 0.5

t_time = 1

dataset = 'dblp'
network = 'data'
image_name = dataset + "_dimension_150"

node_size_set = {"acm" : 11246, "amazon" : 13114, "yelp" : 3913, "dblp" : 37791 , "aminer" : 51012}
type_size_set = {"acm" : 3, "amazon" : 4, "yelp" : 5, "dblp" : 4, "aminer" : 4}

walk_lens = {"yelp" : 60, "dblp" : 60, "acm" : 70, "amazon" : 30, "aminer" : 60}
restart = {"yelp" : 0.95, "dblp" : 0.2, "acm" : 0.7, "amazon" : 0.5, "aminer" : 0.3}
top_ks = {"yelp" : [6, 6, 2, 2, 2], "dblp" : [4, 4, 2, 6], "acm" : [4, 6, 2], "amazon" : [6, 3, 4, 20], "aminer" : [6, 6, 6, 6]}
head_set = {"yelp": 5, "dblp" : 6, "acm" : 9, "aminer" : 4}
#
# walk_lens = {"yelp" : 60, "dblp" : 70, "acm" : 70, "amazon" : 300, "aminer" : 30}
# restart = {"yelp" : 0.5, "dblp" : 0.5, "acm" : 0.2, "amazon" : 0.5, "aminer" : 0.4}
# top_ks = {"yelp" : [6, 6, 2, 2, 2], "dblp" : [4, 4, 2, 6], "acm" : [4, 6, 2], "amazon" : [6, 3, 4, 20], "aminer" : [6, 6, 6, 6]}


node_size = node_size_set[dataset]
t_size = type_size_set[dataset] #节点类型的数目
r_size = (t_size - 1) * 2 #关系的数目
d_epoch = 15 #每次判别器迭代次数
g_epoch = 7#每次生成器迭代次数
n_emb = 150 #嵌入大小

walk_len = walk_lens[dataset]
restart_ratio = restart[dataset]
top_k = top_ks[dataset]
n_heads = head_set[dataset]
walk_NUM = 40

