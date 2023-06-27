import tensorflow as tf
import config

class Generator():
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init):
        self.n_node = n_node #节点数量
        self.n_relation = n_relation #关系数量
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]  #嵌入维度

        #with tf.variable_scope('generator'):
        self.node_embedding_matrix = tf.get_variable(name = "gen_node_embedding",
                                                     shape = self.node_emd_init.shape,
                                                     initializer = tf.constant_initializer(self.node_emd_init),
                                                     trainable = True) #常数，形状与初始化相同
        #eu
        # ——————————————————可修改1——————————————————
        self.relation_embedding_matrix = tf.get_variable(name = "gen_relation_embedding",
                                                         shape = [self.n_relation, self.emd_dim, self.emd_dim],
                                                         # initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                         initializer=tf.random_normal_initializer(0, 1),
                                                         trainable = True) #使得每一层的梯度相差不大
        #Mr

        self.gen_w_1 = tf.get_variable(name = 'gen_w',
                                       shape = [self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_1 = tf.get_variable(name = 'gen_b',
                                       shape = [self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_w_2 = tf.get_variable(name = 'gen_w_2',
                                       shape = [self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_2 = tf.get_variable(name = 'gen_b_2',
                                       shape = [self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        #self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id =  tf.placeholder(tf.int32, shape = [None])
        self.relation_id = tf.placeholder(tf.int32, shape = [None])
        self.node_neighbor_id = tf.placeholder(tf.int32, shape = [None])
        self.noise_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])  #σ2I

        self.dis_node_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])
        self.dis_relation_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim, self.emd_dim])
        self.dis_node_neighbor_embedding = tf.placeholder(tf.float32, shape=[None, self.emd_dim])

        self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_id)
        self.relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.relation_id)
        self.node_neighbor_embedding_f = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_neighbor_id)

        self.node_neighbor_embedding = self.generate_node(self.node_embedding, self.relation_embedding, self.noise_embedding)
        self.node_neighbor_embedding_j = self.generate_node(self.node_neighbor_embedding_f, self.relation_embedding, self.noise_embedding) #错误

        # ——————————————————cos_sim——————————————————————
    #
    #     #故地重游 loss function   dis即判别器的嵌入，即拿着dis作为标准
    #     t = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_embedding, 1), self.dis_relation_embedding), [-1, self.emd_dim])
    #     # self.score = tf.reduce_sum(tf.multiply(t, self.node_neighbor_embedding), axis = 1)
    #
    #     self.score = self.get_similarity_score_2(t, self.node_neighbor_embedding)
    #     self.loss_1 = tf.reduce_mean(tf.square(tf.subtract(self.score, 1)))
    #
    #
    #
    #     t1 = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_neighbor_embedding, 1), self.dis_relation_embedding),[-1, self.emd_dim])
    #     # self.score_2 = self.tf_cosine_distance(t1, self.node_neighbor_embedding_j)
    #     self.score_2 = self.get_similarity_score_2(t1, self.node_neighbor_embedding_j)
    #     self.loss_2 = tf.reduce_mean(tf.square(tf.subtract(self.score_2, 1)))
    #
    #     self.loss = self.loss_1 + self.loss_2 + config.lambda_gen * ((tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(self.relation_embedding) + tf.nn.l2_loss(self.gen_w_1)) + tf.nn.l2_loss(self.gen_b_1))
    #
    #     optimizer = tf.train.AdamOptimizer(config.lr_gen)
    #     # optimizer = tf.train.GradientDescentOptimizer(config.lr_gen)
    #     #optimizer = tf.train.RMSPropOptimizer(config.lr_gen)
    #     self.g_updates = optimizer.minimize(self.loss)
    #
    # def generate_node(self, node_embedding, relation_embedding, noise_embedding):
    #     #node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
    #     #relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)
    #
    #
    #     input = tf.reshape(tf.matmul(tf.expand_dims(node_embedding, 1), relation_embedding), [-1, self.emd_dim])
    #     #input = tf.concat([input, noise_embedding], axis = 1)
    #
    #     input = input + noise_embedding # 论文p4/ 7 高斯分布
    #
    #     output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)  #p4/ 8 MLP
    #     #input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
    #     # output = tf.nn.leaky_relu(tf.matmul(output, self.gen_w_2) + self.gen_b_2)
    #     #output = node_embedding + relation_embedding + noise_embedding
    #
    #     return output
    #
    #
    # def get_similarity_score(self, node_st_emb, relation_emb, node_ed_emb):
    #     t1 = tf.reshape(tf.matmul(tf.expand_dims(node_st_emb, 1), relation_emb), [-1, self.emd_dim])
    #     t2 = tf.reshape(tf.matmul(tf.expand_dims(node_ed_emb, 1), relation_emb), [-1, self.emd_dim])
    #     score = self.tf_cosine_distance(t1, t2)
    #     return score
    #
    # def get_similarity_score_2(self, x3, x4):
    #
    #     self.x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=1))
    #     self.x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=1))
    #     # 内积
    #     self.x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=1)
    #     cosin = self.x3_x4 / (self.x3_norm * self.x4_norm)
    #     # cosin1 = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
    #     return cosin


    # --------------------L2_sim----------------------


        t = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_embedding, 1), self.dis_relation_embedding),
                       [-1, self.emd_dim])
        self.score = self.get_similarity_score_2(t, self.node_neighbor_embedding)
        self.loss_1 = tf.reduce_mean(tf.square(tf.subtract(self.score, 0)))



        t1 = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_neighbor_embedding, 1), self.dis_relation_embedding),
                        [-1, self.emd_dim])
        # self.score_2 = self.tf_cosine_distance(t1, self.node_neighbor_embedding_j)
        self.score_2 = self.get_similarity_score_2(t1, self.node_neighbor_embedding_j)
        self.loss_2 = tf.reduce_mean(tf.square(tf.subtract(self.score_2, 0)))

        # ------------increase-----------

        t1 = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_embedding, 1), self.dis_relation_embedding),
                        [-1, self.emd_dim])
        t2 = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_neighbor_embedding, 1), self.dis_relation_embedding),
                        [-1, self.emd_dim])
        self.neg_score = self.get_similarity_score_2(t1, t2)

        self.neg_score_4 = self.get_similarity_score_2(self.node_neighbor_embedding,
                                                       self.node_neighbor_embedding_j)

        # self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.neg_score_2), logits=self.neg_score_2))
        self.neg_loss_4 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_4, self.neg_score)))

        self.loss = self.loss_1 + self.loss_2 + config.lambda_gen * ((tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(self.relation_embedding) + tf.nn.l2_loss(self.gen_w_1)) + tf.nn.l2_loss(self.gen_b_1))


        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)


    def generate_node(self, node_embedding, relation_embedding, noise_embedding):
        # node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
        # relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)

        input = tf.reshape(tf.matmul(tf.expand_dims(node_embedding, 1), relation_embedding), [-1, self.emd_dim])
        # input = tf.concat([input, noise_embedding], axis = 1)

        input = input + noise_embedding  # 论文p4/ 7 高斯分布

        output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)  # p4/ 8 MLP
        # input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
        # output = tf.nn.leaky_relu(tf.matmul(output, self.gen_w_2) + self.gen_b_2)
        # output = node_embedding + relation_embedding + noise_embedding

        return output


    def get_similarity_score_2(self, node_emb_g, node_emb_d):
        t = node_emb_g - node_emb_d
        score = tf.sqrt(tf.reduce_sum(tf.square(t), axis=1))
        return score





    #     # _____________________点积+sigmoid+交叉熵_____________________
    #
    #     # 故地重游 loss function   dis即判别器的嵌入，即拿着dis作为标准
    #
    #     # ——————————————————————here j节点的相似性——————————————————————————
    #     t = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_neighbor_embedding, 1), self.dis_relation_embedding),
    #                    [-1, self.emd_dim])
    #     self.score_1 = tf.reduce_sum(tf.multiply(t, self.node_neighbor_embedding_j), axis=1)
    #     self.loss_1 = tf.reduce_sum(
    #         tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_1) * (1.0 - config.label_smooth),
    #                                                 logits=self.score_1))
    #
    #     # ——————————————————————here i节点的相似性——————————————————————————
    #     t = tf.reshape(tf.matmul(tf.expand_dims(self.dis_node_embedding, 1), self.dis_relation_embedding),
    #                    [-1, self.emd_dim])
    #     self.score_2 = tf.reduce_sum(tf.multiply(t, self.node_neighbor_embedding), axis=1)
    #     self.loss_2 = tf.reduce_sum(
    #         tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score_2) * (1.0 - config.label_smooth),
    #                                                 logits=self.score_2))
    #
    #     self.loss = self.loss_1 + self.loss_2 + config.lambda_gen * (
    #                 (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(
    #                     self.relation_embedding) + tf.nn.l2_loss(self.gen_w_1)) + tf.nn.l2_loss(self.gen_b_1))
    #
    #     optimizer = tf.train.AdamOptimizer(config.lr_gen)
    #     # optimizer = tf.train.RMSPropOptimizer(config.lr_gen)
    #     self.g_updates = optimizer.minimize(self.loss)
    #
    #
    # def generate_node(self, node_embedding, relation_embedding, noise_embedding):
    #     # node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
    #     # relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)
    #
    #     input = tf.reshape(tf.matmul(tf.expand_dims(node_embedding, 1), relation_embedding), [-1, self.emd_dim])
    #     # input = tf.concat([input, noise_embedding], axis = 1)
    #
    #     input = input + noise_embedding  # 论文p4/ 7 高斯分布
    #
    #     output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)  # p4/ 8 MLP
    #     # input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
    #     # output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_2) + self.gen_b_2)
    #     # output = node_embedding + relation_embedding + noise_embedding
    #
    #     return output
    #
    #
    # def get_similarity_sorce(self, node_st_emb, relation_emb, node_ed_emb):
    #     t1 = tf.reshape(tf.matmul(tf.expand_dims(node_st_emb, 1), relation_emb), [-1, self.emd_dim])
    #     t2 = tf.reshape(tf.matmul(tf.expand_dims(node_ed_emb, 1), relation_emb), [-1, self.emd_dim])
    #     score = tf.reduce_sum(tf.multiply(t1, t2), axis=1)
    #     return score


