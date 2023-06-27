import tensorflow as tf
import config

class Discriminator():
    def __init__(self, n_node, n_relation, node_emd_init, relation_emd_init):
        self.n_node = n_node
        self.n_relation = n_relation
        self.node_emd_init = node_emd_init
        self.relation_emd_init = relation_emd_init
        self.emd_dim = node_emd_init.shape[1]



        #with tf.variable_scope('disciminator'):
        self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding',
                                                     shape = self.node_emd_init.shape,
                                                     initializer =tf.constant_initializer(self.node_emd_init),
                                                     trainable = True)

        self.relation_embedding_matrix = tf.get_variable(name = 'dis_relation_embedding',
                                                         shape = [self.n_relation, self.emd_dim, self.emd_dim],
                                                         # initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                         initializer = tf.random_normal_initializer(0, 1),
                                                         trainable = True)
        # self.r_b = tf.get_variable(name = 'dis_relation_embedding_b',
        #                                                  shape = [self.n_relation, self.emd_dim],
        #                                                  # initializer = tf.contrib.layers.xavier_initializer(uniform = False),
        #                                                  initializer = tf.random_normal_initializer(0, 1),
        #                                                  trainable = True)
        #Mr


        self.alaph = tf.placeholder(tf.float32, shape=[None])
        self.pos_node_id = tf.placeholder(tf.int32, shape = [None])
        self.pos_relation_id = tf.placeholder(tf.int32, shape = [None])
        self.pos_node_neighbor_id = tf.placeholder(tf.int32, shape = [None])

        self.neg_node_id_1 = tf.placeholder(tf.int32, shape = [None])
        self.neg_relation_id_1 = tf.placeholder(tf.int32, shape = [None])
        self.neg_node_neighbor_id_1 = tf.placeholder(tf.int32, shape = [None])

        self.neg_node_id_2 = tf.placeholder(tf.int32, shape = [None])
        self.neg_relation_id_2 = tf.placeholder(tf.int32, shape = [None])
        self.neg_node_neighbor_id_2 = tf.placeholder(tf.int32, shape = [None])

        self.node_fake_neighbor_embedding = tf.placeholder(tf.float32, shape = [None, self.emd_dim])
        self.node_fake_neighbor_embedding_j = tf.placeholder(tf.float32, shape = [None, self.emd_dim])

        self.pos_node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_id)
        self.pos_node_neighbor_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.pos_node_neighbor_id)
        self.pos_relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.pos_relation_id)

        self.neg_node_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_1)
        self.neg_node_neighbor_embedding_1 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_neighbor_id_1)
        self.neg_relation_embedding_1 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_1)

        self.neg_node_embedding_2 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_id_2)
        self.neg_relation_embedding_2 = tf.nn.embedding_lookup(self.relation_embedding_matrix, self.neg_relation_id_2)
        self.neg_node_neighbor_embedding_2 = tf.nn.embedding_lookup(self.node_embedding_matrix, self.neg_node_neighbor_id_2)

    # ---------------cos_sim--------------------

    #
    #     #pos loss
    #     # t = tf.reshape(tf.matmul(tf.expand_dims(self.pos_node_embedding, 1), self.pos_relation_embedding), [-1, self.emd_dim])
    #     # self.pos_score = tf.reduce_sum(tf.multiply(t, self.pos_node_neighbor_embedding), axis = 1)   #论文p4 / 2
    #
    #
    #     self.pos_score = self.get_similarity_score(self.pos_node_embedding, self.pos_relation_embedding,self.pos_node_neighbor_embedding)
    #     self.pos_loss =  tf.reduce_mean(tf.square(tf.subtract(self.pos_score, 1)))
    #
    #     # self.pos_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.pos_score), logits=self.pos_score))#一致
    #
    #     #neg loss_1
    #     # t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_1, 1), self.neg_relation_embedding_1), [-1, self.emd_dim])
    #     # self.neg_score_1 = tf.reduce_sum(tf.multiply(t, self.neg_node_neighbor_embedding_1), axis = 1)
    #
    #     self.neg_score_1 = self.get_similarity_score(self.neg_node_embedding_1, self.neg_relation_embedding_1, self.neg_node_neighbor_embedding_1)
    #     # self.neg_loss_1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.neg_score_1), logits=self.neg_score_1))
    #     self.neg_loss_1 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_1, -1)))
    #     #neg loss_2
    #
    #     # self.neg_score_2 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding), axis = 1)
    #     # self.neg_score_2 = self.get_similarity_score(self.neg_node_embedding_2, self.neg_relation_embedding_2, self.node_fake_neighbor_embedding)
    #
    #     # self.neg_score_2 = self.get_similarity_score_2(t, self.node_fake_neighbor_embedding)
    #     # # self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.neg_score_2), logits=self.neg_score_2))
    #     # self.neg_loss_2 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_2, 30)))
    #     t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2), [-1, self.emd_dim])
    #     self.neg_score_2 = self.get_similarity_score_2(t, self.node_fake_neighbor_embedding_j)
    #     # self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.neg_score_2), logits=self.neg_score_2))
    #     self.neg_loss_2 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_2, -1)))
    #
    #
    #     t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_neighbor_embedding_2, 1), self.neg_relation_embedding_2),[-1, self.emd_dim])
    #     self.neg_score_3 = self.get_similarity_score_2(t, self.node_fake_neighbor_embedding)
    #     # self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.neg_score_2), logits=self.neg_score_2))
    #     self.neg_loss_3 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_3, -1)))
    #
    #
    #
    #     t1 = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2),[-1, self.emd_dim])
    #     t2 = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_neighbor_embedding_2, 1), self.neg_relation_embedding_2),[-1, self.emd_dim])
    #     self.neg_score = self.get_similarity_score_2(t1, t2)
    #
    #     self.neg_score_4 = self.get_similarity_score_2(self.node_fake_neighbor_embedding, self.node_fake_neighbor_embedding_j)
    #
    #
    #     # self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.neg_score_2), logits=self.neg_score_2))
    #     self.neg_loss_4 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_4, self.neg_score)))
    #
    #     # self.loss = self.pos_loss + self.neg_loss_1 +  self.neg_loss_3
    #     self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2 + self.neg_loss_3 + self.neg_loss_4
    #     # + \
    #     #     config.lambda_dis * (tf.nn.l2_loss(self.pos_node_embedding) + tf.nn.l2_loss(self.pos_relation_embedding) + \
    #     #                          tf.nn.l2_loss(self.neg_node_embedding_1) + tf.nn.l2_loss(
    #     #             self.neg_relation_embedding_1) + \
    #     #                          tf.nn.l2_loss(self.neg_node_embedding_2) + tf.nn.l2_loss(
    #     #             self.neg_relation_embedding_2))
    #
    #     optimizer = tf.train.AdamOptimizer(config.lr_dis)
    #     # optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
    #     #optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
    #     #optimizer = tf.train.RMSPropOptimizer(config.lr_dis)
    #     self.d_updates = optimizer.minimize(self.loss)
    #     #self.reward = tf.log(1 + tf.exp(tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)))
    #
    # def get_similarity_score(self, node_st_emb, relation_emb, node_ed_emb):
    #     t1 = tf.reshape(tf.matmul(tf.expand_dims(node_st_emb, 1), relation_emb), [-1, self.emd_dim])
    #     t2 = tf.reshape(tf.matmul(tf.expand_dims(node_ed_emb, 1), relation_emb), [-1, self.emd_dim])
    #     score = self.get_similarity_score_2(t1, t2)
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

    # _______________L2_sim_________________


        # ----------------change 2----------------
        self.pos_score = self.get_similarity_score(self.pos_node_embedding, self.pos_relation_embedding,
                                                   self.pos_node_neighbor_embedding)
        self.pos_loss = tf.reduce_mean(tf.square(tf.subtract(self.pos_score, self.alaph)))


        self.neg_score_1 = self.get_similarity_score(self.neg_node_embedding_1, self.neg_relation_embedding_1, self.neg_node_neighbor_embedding_1)
        self.neg_loss_1 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_1, config.inf)))



        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2),
                       [-1, self.emd_dim])
        self.neg_score_2 = self.get_similarity_score_2(t, self.node_fake_neighbor_embedding_j)
        self.neg_loss_2 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_2, config.inf)))


        t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_neighbor_embedding_2, 1), self.neg_relation_embedding_2),
                       [-1, self.emd_dim])
        self.neg_score_3 = self.get_similarity_score_2(t, self.node_fake_neighbor_embedding)
        self.neg_loss_3 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_3, config.inf)))



        # t1 = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2),
        #                 [-1, self.emd_dim])
        # t2 = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_neighbor_embedding_2, 1), self.neg_relation_embedding_2),
        #                 [-1, self.emd_dim])
        # self.neg_score = self.get_similarity_score_2(t1, t2)
        #
        # self.neg_score_4 = self.get_similarity_score_2(self.node_fake_neighbor_embedding,
        #                                                self.node_fake_neighbor_embedding_j)
        #
        # # self.neg_loss_2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.neg_score_2), logits=self.neg_score_2))
        # self.neg_loss_4 = tf.reduce_mean(tf.square(tf.subtract(self.neg_score_4, self.neg_score)))

        self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2 + self.neg_loss_3

        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)




    def get_similarity_score(self, node_st_emb, relation_emb, node_ed_emb):
        self.t1 = tf.reshape(tf.matmul(tf.expand_dims(node_st_emb, 1), relation_emb), [-1, self.emd_dim])
        self.t2 = tf.reshape(tf.matmul(tf.expand_dims(node_ed_emb, 1), relation_emb), [-1, self.emd_dim])
        self.t = self.t1 - self.t2
        score = tf.sqrt(tf.reduce_sum(tf.square(self.t), axis=1))
        return score

    def get_similarity_score_2(self, node_emb_g, node_emb_d):
        t = node_emb_g - node_emb_d
        score = tf.sqrt(tf.reduce_sum(tf.square(t), axis=1))
        return score


    # # -----------------点积-------------------
    #
    #     self.pos_score = self.get_similarity_sorce(self.pos_node_embedding, self.pos_relation_embedding,
    #                                                self.pos_node_neighbor_embedding)
    #     self.pos_loss = tf.reduce_sum(
    #         tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pos_score), logits=self.pos_score))  # 一致
    #
    #     self.neg_score_1 = self.get_similarity_sorce(self.neg_node_embedding_1, self.neg_relation_embedding_1,
    #                                                  self.neg_node_neighbor_embedding_1)
    #     self.neg_loss_1 = tf.reduce_sum(
    #         tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_1), logits=self.neg_score_1))
    #
    #     # neg loss_2
    #     # t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2), [-1, self.emd_dim])
    #     # self.neg_score_2 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding), axis = 1)
    #
    #     # ____________________self.neg_node_neighbor_embedding_2 为 节点j _______________________
    #     t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_neighbor_embedding_2, 1), self.neg_relation_embedding_2),
    #                    [-1, self.emd_dim])
    #     self.neg_score_2 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding), axis=1)
    #
    #     self.neg_loss_2 = tf.reduce_sum(
    #         tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_2), logits=self.neg_score_2))
    #
    #     # ------------------D中的i与G中j--------------------
    #     t = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2),
    #                    [-1, self.emd_dim])
    #     self.neg_score_3 = tf.reduce_sum(tf.multiply(t, self.node_fake_neighbor_embedding_j), axis=1)
    #
    #     self.neg_loss_3 = tf.reduce_sum(
    #         tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_3), logits=self.neg_score_3))
    #
    #     # --------------------------两种分数相比-------------------------
    #     # t1 = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_embedding_2, 1), self.neg_relation_embedding_2),
    #     #                [-1, self.emd_dim])
    #     #
    #     # t2 = tf.reshape(tf.matmul(tf.expand_dims(self.neg_node_neighbor_embedding_2, 1), self.neg_relation_embedding_2),
    #     #                [-1, self.emd_dim])
    #     # dis_t = tf.reduce_sum(tf.multiply(t1, t2), axis=1)
    #     #
    #     # gen_t = tf.reduce_sum(tf.multiply(self.node_fake_neighbor_embedding, self.node_fake_neighbor_embedding_j), axis=1)
    #     #
    #     # self.neg_score_4 = tf.reduce_sum(tf.multiply(dis_t, gen_t), axis=1)
    #     # self.neg_loss_4 = tf.reduce_sum(
    #     #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_score_4), logits=self.neg_score_4))
    #
    #     self.loss = self.pos_loss + self.neg_loss_1 + self.neg_loss_2 + self.neg_loss_3
    #     # + \
    #     #     config.lambda_dis * (tf.nn.l2_loss(self.pos_node_embedding) + tf.nn.l2_loss(self.pos_relation_embedding) + \
    #     #                          tf.nn.l2_loss(self.neg_node_embedding_1) + tf.nn.l2_loss(
    #     #             self.neg_relation_embedding_1) + \
    #     #                          tf.nn.l2_loss(self.neg_node_embedding_2) + tf.nn.l2_loss(
    #     #             self.neg_relation_embedding_2))
    #
    #     optimizer = tf.train.AdamOptimizer(config.lr_dis)
    #     # optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
    #     # optimizer = tf.train.RMSPropOptimizer(config.lr_dis)
    #     self.d_updates = optimizer.minimize(self.loss)
    #     # self.reward = tf.log(1 + tf.exp(tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)))
    #
    #
    # def get_similarity_sorce(self, node_st_emb, relation_emb, node_ed_emb):
    #     t1 = tf.reshape(tf.matmul(tf.expand_dims(node_st_emb, 1), relation_emb), [-1, self.emd_dim])
    #     t2 = tf.reshape(tf.matmul(tf.expand_dims(node_ed_emb, 1), relation_emb), [-1, self.emd_dim])
    #     score = tf.reduce_sum(tf.multiply(t1, t2), axis=1)
    #     return score
    #





