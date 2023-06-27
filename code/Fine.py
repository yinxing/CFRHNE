import os
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import tensorflow as tf
import config
import generator
import discriminator
import utils
import time
import numpy as np
from dblp_evaluation import DBLP_evaluation
from yelp_evaluation import Yelp_evaluation
from aminer_evaluation import Aminer_evaluation
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from tensorflow.python import debug as tfdbg
from random_walk import random_walk
import copy
from acm_evaluation import ACM_evaluation


def seed_tensorflow(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

seed_tensorflow()

class Model():
    def __init__(self):

        t = time.time()
        print("reading graph...")
        self.n_node, self.n_relation, self.graph, self.g_type, self.norm_gt, self.m_type, self.size, self.neg_g, self.r_adj_g, self.c_adj_g = utils.read_graph(config.graph_filename) # n.node， n_relation 为节点，关系的数量， graph即图 格式map<u, map<r, v>>
        self.node_list = list(self.graph.keys())#range(0, self.n_node) 出度不为0的节点list
        print('[%.2f] reading graph finished. #node = %d #relation = %d' % (time.time() - t, self.n_node, self.n_relation))
        self.t_relation = np.array(
            [[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])


        t = time.time()
        print("read initial embeddings...") #预处理嵌入 ?, 将嵌入信息输入矩阵中
        # self.node_embed_init_d = utils.read_embeddings(filename=config.pretrain_node_emb_filename_d,
        #                                                n_node=self.n_node,
        #                                                n_embed=config.n_emb)
        # self.node_embed_init_g = utils.read_embeddings(filename=config.pretrain_node_emb_filename_g,
        #                                                n_node=self.n_node,
        #                                                n_embed=config.n_emb)

        #
        # self.node_embed_init_d = utils.read_new_embeddings('../pre_train/node_clustering/dblp_pre_train.emb'.format(cnt))
        # self.node_embed_init_g = utils.read_new_embeddings('../pre_train/node_clustering/dblp_pre_train.emb'.format(cnt))
        #
                                                                            

        self.node_embed_init_d = utils.read_new_embeddings('../dimension_emb/' + config.dataset + '/yelp_dimension_64_14.txt')
        self.node_embed_init_g = utils.read_new_embeddings('../dimension_emb/' + config.dataset + '/yelp_dimension_64_14.txt')

        #
        # self.rel_embed_init_d = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_d,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        # self.rel_embed_init_g = utils.read_embeddings(filename=config.pretrain_rel_emb_filename_g,
        #                                              n_node=self.n_node,
        #                                              n_embed=config.n_emb)
        print("[%.2f] read initial embeddings finished." % (time.time() - t))

        print( "build GAN model...")
        tf.reset_default_graph()
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()  #建立生成器与判别器

        self.latest_checkpoint = tf.train.latest_checkpoint(config.model_log)
        self.saver = tf.train.Saver() #记得调一下

        self.dblp_evaluation = DBLP_evaluation()
        self.yelp_evaluation = Yelp_evaluation()
        self.acm_evaluation = ACM_evaluation()
        self.aminer_evaluation = Aminer_evaluation()
        # self.aminer_evaluation = Aminer_evaluation()
        # self.yelp_evaluation.evaluate_business_classification(self.node_embed_init_g)
        # exit(0)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session(config = self.config)
        # self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess)  # 被调试器封装的会话
        # self.sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
        self.sess.run(self.init_op)

        self.show_config()

    def show_config(self):
        print('--------------------')
        print('Model config : ')
        print('img_name = ', config.image_name)
        print('dataset = ', config.dataset)
        print('batch_size = ', config.batch_size)
        print('lambda_gen = ', config.lambda_gen)
        print('lambda_dis = ', config.lambda_dis)
        print('n_sample = ', config.n_sample)
        print('lr_gen = ', config.lr_gen)
        print('lr_dis = ', config.lr_dis)
        print('n_epoch = ', config.n_epoch)
        print('d_epoch = ', config.d_epoch)
        print('g_epoch = ', config.g_epoch)
        print('n_emb = ', config.n_emb)
        print('sig = ', config.sig)
        print('label smooth = ', config.label_smooth)
        print('--------------------')

    def build_generator(self):
        #with tf.variable_scope("generator"):
        self.generator = generator.Generator(n_node = self.n_node,
                                             n_relation = self.n_relation,
                                             node_emd_init = self.node_embed_init_g,
                                             relation_emd_init = None)
    def build_discriminator(self):
        #with tf.variable_scope("discriminator"):
        self.discriminator = discriminator.Discriminator(n_node = self.n_node,
                                                         n_relation = self.n_relation,
                                                         node_emd_init = self.node_embed_init_d,
                                                         relation_emd_init = None)

    def train(self, k1, flag, k2, img_name, cnt):

        gen_nmi_data = []
        dis_nmi_data = []
        gen_mi_f1s = []
        dis_mi_f1s = []
        gen_ma_f1s = []
        dis_ma_f1s = []
        gen_aucs = []
        gen_accs = []
        gen_f1s = []
        dis_aucs = []
        dis_accs = []
        dis_f1s = []
        epoch_datas = []

        dis_losses = []
        gen_losses = []
        pos_losses = []
        neg_1_losses = []
        neg_2_losses = []

        print('start traning...')
        for epoch in range(config.n_epoch):
            if(epoch % config.change == 0):
                flag = 1 - flag

            if(flag):
                k = k2
            else:
                k = k1
            print('epoch %d' % epoch)
            t = time.time()

            one_epoch_gen_loss = 0.0
            one_epoch_dis_loss = 0.0
            one_epoch_batch_num = 0.0

            #D-step
            #t1 = time.time()
            for d_epoch in range(config.d_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_dis_loss = 0.0
                one_epoch_pos_loss = 0.0
                one_epoch_neg_loss_1 = 0.0
                one_epoch_neg_loss_2 = 0.0

                for index in tqdm(range(len(self.node_list) // config.batch_size)):


                    pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, neg_node_neighbor_ids_2, node_fake_neighbor_embedding, node_fake_neighbor_embedding_j = self.prepare_data_for_d(index, flag)


                    node_1, node_r_1, node_n_1,  _, dis_loss, pos_loss, neg_loss_1, neg_loss_2, neg_score_1, dis_node_embs, pos_score  = \
                        self.sess.run([self.discriminator.neg_node_embedding_1, self.discriminator.neg_relation_embedding_1, self.discriminator.neg_node_neighbor_embedding_1, self.discriminator.d_updates, self.discriminator.loss, self.discriminator.pos_loss, self.discriminator.neg_loss_1, self.discriminator.neg_loss_3, self.discriminator.neg_score_1, self.discriminator.node_embedding_matrix, self.discriminator.pos_score],
                                                 feed_dict = {
                                                              self.discriminator.alaph : np.array([k]),
                                                              self.discriminator.pos_node_id : np.array(pos_node_ids),
                                                              self.discriminator.pos_relation_id : np.array(pos_relation_ids),
                                                              self.discriminator.pos_node_neighbor_id : np.array(pos_node_neighbor_ids),
                                                              self.discriminator.neg_node_id_1 : np.array(neg_node_ids_1),
                                                              self.discriminator.neg_relation_id_1 : np.array(neg_relation_ids_1),
                                                              self.discriminator.neg_node_neighbor_id_1 : np.array(neg_node_neighbor_ids_1),
                                                              self.discriminator.neg_node_id_2 : np.array(neg_node_ids_2),
                                                              self.discriminator.neg_relation_id_2 : np.array(neg_relation_ids_2),
                                                              self.discriminator.neg_node_neighbor_id_2 : np.array(neg_node_neighbor_ids_2),
                                                              self.discriminator.node_fake_neighbor_embedding : np.array(node_fake_neighbor_embedding),
                                                              self.discriminator.node_fake_neighbor_embedding_j : np.array(node_fake_neighbor_embedding_j)})

                    one_epoch_dis_loss += dis_loss
                    one_epoch_pos_loss += pos_loss
                    one_epoch_neg_loss_1 += neg_loss_1
                    one_epoch_neg_loss_2 += neg_loss_2



            #G-step

            # ----------------轮数修改--------------------


            for g_epoch in range(config.g_epoch):
                np.random.shuffle(self.node_list)
                one_epoch_gen_loss = 0.0

                for index in tqdm(range(len(self.node_list) // config.batch_size)):

                    gen_node_ids, gen_relation_ids, gen_node_neighbor_ids, gen_noise_embedding, gen_dis_node_embedding, gen_dis_relation_embedding, gen_dis_node_neighbor_embedding = self.prepare_data_for_g(index, flag)
                    t2 = time.time()

                    _, gen_loss, sorces = self.sess.run([self.generator.g_updates, self.generator.loss, self.generator.score],
                                                 feed_dict = {self.generator.node_id :  np.array(gen_node_ids),
                                                              self.generator.relation_id :  np.array(gen_relation_ids),
                                                              self.generator.node_neighbor_id : np.array(gen_node_neighbor_ids),
                                                              self.generator.noise_embedding : np.array(gen_noise_embedding),
                                                              self.generator.dis_node_embedding : np.array(gen_dis_node_embedding),
                                                              self.generator.dis_relation_embedding : np.array(gen_dis_relation_embedding),
                                                              self.generator.dis_node_neighbor_embedding : np.array(gen_dis_node_neighbor_embedding)})

                    one_epoch_gen_loss += gen_loss

                print(sorces)
            one_epoch_batch_num = len(self.node_list) / config.batch_size


            print(one_epoch_batch_num)
            print(one_epoch_neg_loss_2)
            print(one_epoch_neg_loss_1)
            print(one_epoch_pos_loss)
            print(one_epoch_dis_loss)
            print(one_epoch_gen_loss)


            #print() t2 - t1
            #exit(
            print('gen loss = %.4f, dis loss = %.4f pos loss = %.4f neg loss-1 = %.4f neg loss-2 = %.4f' % \
                  (one_epoch_gen_loss / one_epoch_batch_num, one_epoch_dis_loss / one_epoch_batch_num,
                   one_epoch_pos_loss / one_epoch_batch_num, one_epoch_neg_loss_1 / one_epoch_batch_num,
                   one_epoch_neg_loss_2 / one_epoch_batch_num))

            gen_losses.append(one_epoch_gen_loss / one_epoch_batch_num)
            dis_losses.append(one_epoch_dis_loss / one_epoch_batch_num)
            pos_losses.append(one_epoch_pos_loss / one_epoch_batch_num)
            neg_1_losses.append(one_epoch_neg_loss_1 / one_epoch_batch_num)
            neg_2_losses.append(one_epoch_neg_loss_2 / one_epoch_batch_num)


            if config.dataset == 'dblp':
                gen_nmi, dis_nmi = self.evaluate_author_cluster()
                aucs, accs, f1s = self.evaluate_dblp_link_prediction()
                micro_f1s, macro_f1s = self.evaluate_author_classification()
                gen_fi, dis_fi = micro_f1s
                gen_fa, dis_fa = macro_f1s
                gen_auc, dis_auc = aucs
                gen_acc, dis_acc = accs
                gen_f1, dis_f1 = f1s

                gen_nmi_data.append(gen_nmi)
                dis_nmi_data.append(dis_nmi)
                gen_mi_f1s.append(gen_fi)
                dis_mi_f1s.append(dis_fi)
                gen_ma_f1s.append(gen_fa)
                dis_ma_f1s.append(dis_fa)
                gen_aucs.append(gen_auc)
                gen_accs.append(gen_acc)
                gen_f1s.append(gen_f1)
                dis_aucs.append(dis_auc)
                dis_accs.append(dis_acc)
                dis_f1s.append(dis_f1)
                epoch_datas.append(epoch)
                # x.append(epoch)
                # y.append(dis_nmi)
                # y1.append(gen_nmi)
                print('---------------------cluster-----------------------')
                print('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
                print('-----------------classification-----------------------')
                print('Gen micro_f1s = %.4f Dis micro_f1s = %.4f' % (gen_fi, dis_fi))
                print('Gen macro_f1s = %.4f Dis macro_f1s = %.4f' % (gen_fa, dis_fa))

                print('-----------------link_prediction---------------------')
                print('Gen auc = %.4f Dis auc = %.4f' % (gen_auc, dis_auc))
                print('Gen acc = %.4f Dis acc = %.4f' % (gen_acc, dis_acc))
                print('Gen f1 = %.4f Dis f1 = %.4f' % (gen_f1, dis_f1))
            elif config.dataset == 'yelp':
                gen_nmi, dis_nmi = self.evaluate_business_cluster()
                aucs, accs, f1s = self.evaluate_yelp_link_prediction()
                micro_f1s, macro_f1s = self.evaluate_business_classification()
                gen_fi, dis_fi = micro_f1s
                gen_fa, dis_fa = macro_f1s
                gen_auc, dis_auc = aucs
                gen_acc, dis_acc = accs
                gen_f1, dis_f1 = f1s

                gen_nmi_data.append(gen_nmi)
                dis_nmi_data.append(dis_nmi)
                gen_mi_f1s.append(gen_fi)
                dis_mi_f1s.append(dis_fi)
                gen_ma_f1s.append(gen_fa)
                dis_ma_f1s.append(dis_fa)
                gen_aucs.append(gen_auc)
                gen_accs.append(gen_acc)
                gen_f1s.append(gen_f1)
                dis_aucs.append(dis_auc)
                dis_accs.append(dis_acc)
                dis_f1s.append(dis_f1)
                epoch_datas.append(epoch)
                # x.append(epoch)
                # y.append(dis_nmi)
                # y1.append(gen_nmi)
                print('---------------------cluster-----------------------')
                print('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
                print('-----------------classification-----------------------')
                print('Gen micro_f1s = %.4f Dis micro_f1s = %.4f' % (gen_fi, dis_fi))
                print('Gen macro_f1s = %.4f Dis macro_f1s = %.4f' % (gen_fa, dis_fa))

                print('-----------------link_prediction---------------------')
                print('Gen auc = %.4f Dis auc = %.4f' % (gen_auc, dis_auc))
                print('Gen acc = %.4f Dis acc = %.4f' % (gen_acc, dis_acc))
                print('Gen f1 = %.4f Dis f1 = %.4f' % (gen_f1, dis_f1))
                # micro_f1s, macro_f1s = self.evaluate_business_classification()
                # print() 'Gen Micro_f1 = %.4f Dis Micro_f1 = %.4f' %(micro_f1s[0], micro_f1s[1])
                # print() 'Gen Macro_f1 = %.4f Dis Macro_f1 = %.4f' %(macro_f1s[0], macro_f1s[1])

            elif config.dataset == "acm":
                gen_nmi, dis_nmi = self.evaluate_acm_cluster()
                aucs, accs, f1s = self.evaluate_acm_link_prediction()
                micro_f1s, macro_f1s = self.evaluate_acm_classification()
                gen_fi, dis_fi = micro_f1s
                gen_fa, dis_fa = macro_f1s
                gen_auc, dis_auc = aucs
                gen_acc, dis_acc = accs
                gen_f1, dis_f1 = f1s

                gen_nmi_data.append(gen_nmi)
                dis_nmi_data.append(dis_nmi)
                gen_mi_f1s.append(gen_fi)
                dis_mi_f1s.append(dis_fi)
                gen_ma_f1s.append(gen_fa)
                dis_ma_f1s.append(dis_fa)
                gen_aucs.append(gen_auc)
                gen_accs.append(gen_acc)
                gen_f1s.append(gen_f1)
                dis_aucs.append(dis_auc)
                dis_accs.append(dis_acc)
                dis_f1s.append(dis_f1)
                epoch_datas.append(epoch)
                # x.append(epoch)
                # y.append(dis_nmi)
                # y1.append(gen_nmi)
                print('---------------------cluster-----------------------')
                print('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))
                print('-----------------classification-----------------------')
                print('Gen micro_f1s = %.4f Dis micro_f1s = %.4f' % (gen_fi, dis_fi))
                print('Gen macro_f1s = %.4f Dis macro_f1s = %.4f' % (gen_fa, dis_fa))

                print('-----------------link_prediction---------------------')
                print('Gen auc = %.4f Dis auc = %.4f' % (gen_auc, dis_auc))
                print('Gen acc = %.4f Dis acc = %.4f' % (gen_acc, dis_acc))
                print('Gen f1 = %.4f Dis f1 = %.4f' % (gen_f1, dis_f1))



            elif config.dataset == "aminer":

                gen_nmi, dis_nmi = self.evaluate_aminer_cluster()

                aucs, accs, f1s = self.evaluate_aminer_link_prediction()

                micro_f1s, macro_f1s = self.evaluate_aminer_classification()

                gen_fi, dis_fi = micro_f1s

                gen_fa, dis_fa = macro_f1s

                gen_auc, dis_auc = aucs

                gen_acc, dis_acc = accs

                gen_f1, dis_f1 = f1s

                gen_nmi_data.append(gen_nmi)

                dis_nmi_data.append(dis_nmi)

                gen_mi_f1s.append(gen_fi)

                dis_mi_f1s.append(dis_fi)

                gen_ma_f1s.append(gen_fa)

                dis_ma_f1s.append(dis_fa)

                gen_aucs.append(gen_auc)

                gen_accs.append(gen_acc)

                gen_f1s.append(gen_f1)

                dis_aucs.append(dis_auc)

                dis_accs.append(dis_acc)

                dis_f1s.append(dis_f1)

                epoch_datas.append(epoch)

                # x.append(epoch)

                # y.append(dis_nmi)

                # y1.append(gen_nmi)

                print('---------------------cluster-----------------------')

                print('Gen NMI score = %.4f Dis NMI score = %.4f' % (gen_nmi, dis_nmi))

                print('-----------------classification-----------------------')

                print('Gen micro_f1s = %.4f Dis micro_f1s = %.4f' % (gen_fi, dis_fi))

                print('Gen macro_f1s = %.4f Dis macro_f1s = %.4f' % (gen_fa, dis_fa))

                print('-----------------link_prediction---------------------')

                print('Gen auc = %.4f Dis auc = %.4f' % (gen_auc, dis_auc))

                print('Gen acc = %.4f Dis acc = %.4f' % (gen_acc, dis_acc))

                print('Gen f1 = %.4f Dis f1 = %.4f' % (gen_f1, dis_f1))

        self.draw(gen_nmi_data, dis_nmi_data, gen_mi_f1s, dis_mi_f1s, gen_ma_f1s, dis_ma_f1s, gen_aucs, gen_accs,
                  gen_f1s, dis_aucs, dis_accs, dis_f1s, epoch_datas, img_name, cnt)

        self.draw_loss(epoch_datas, gen_losses, dis_losses, pos_losses, neg_1_losses, neg_2_losses, img_name, cnt)
        print("training completes")

    def draw(self, gen_nmi_data, dis_nmi_data, gen_mi_f1s, dis_mi_f1s, gen_ma_f1s, dis_ma_f1s, gen_aucs, gen_accs,
             gen_f1s, dis_aucs, dis_accs, dis_f1s, epoch_datas, suffix, cnt):
        # plot 1:

        ax1 = plt.subplot(2, 2, 1)
        # my_y_ticks = np.arange(0.30, 0.41, 0.005)
        # my_x_ticks = np.arange(0, config.n_epoch, 1)
        # plt.xticks(my_x_ticks)
        # plt.yticks(my_y_ticks)
        plt.plot(epoch_datas, dis_nmi_data, label="dis_nmi")
        plt.plot(epoch_datas, gen_nmi_data, label="gen_nmi")
        plt.title("NMI")
        plt.legend()

        # plot 2:

        plt.subplot(2, 2, 2)
        # my_y_ticks = np.arange(0.7, 0.86, 0.005)
        # my_x_ticks = np.arange(0, config.n_epoch, 1)
        # plt.xticks(my_x_ticks)
        # plt.yticks(my_y_ticks)
        plt.plot(epoch_datas, dis_f1s, label="dis_f1")
        plt.plot(epoch_datas, gen_f1s, label="gen_f1")
        plt.plot(epoch_datas, dis_accs, label="dis_acc")
        plt.plot(epoch_datas, gen_accs, label="gen_acc")
        plt.plot(epoch_datas, dis_aucs, label="dis_auc")
        plt.plot(epoch_datas, gen_aucs, label="gen_auc")
        plt.title("link_prediction")
        plt.legend()

        # plot 3:

        plt.subplot(2, 2, 3)
        # my_y_ticks = np.arange(0.70, 0.82, 0.006)
        # my_x_ticks = np.arange(0, config.n_epoch, 1)
        # plt.xticks(my_x_ticks)
        # plt.yticks(my_y_ticks)
        plt.plot(epoch_datas, dis_mi_f1s, label="dis_micro_f1")
        plt.plot(epoch_datas, gen_mi_f1s, label="gen_micro_f1")
        plt.plot(epoch_datas, dis_ma_f1s, label="dis_macro_f1")
        plt.plot(epoch_datas, gen_ma_f1s, label="gen_macro_f1")
        plt.title("classfication")
        plt.legend()

        plt.suptitle("{}".format(config.image_name))
        plt.savefig('../'+ config.file + '_img/{}.png'.format(config.image_name), dpi=200)
        # plt.show()

    def draw_loss(self, epoch_datas, gen_loss, dis_loss, pos_loss, neg_1_loss, neg_2_loss, suffix, cnt):
        # plot 1:

        ax1 = plt.subplot(2, 3, 1)
        plt.plot(epoch_datas, gen_loss, label="gen_loss")
        plt.title("gen_loss")
        plt.legend()

        # plot 2:

        plt.subplot(2, 3, 2)
        plt.plot(epoch_datas, dis_loss, label="dis_loss")
        plt.title("dis_loss")
        plt.legend()

        # plot 3:

        plt.subplot(2, 3, 3)
        plt.plot(epoch_datas, pos_loss, label="pos_loss")
        plt.title("pos_loss")
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(epoch_datas, neg_1_loss, label="neg_1_loss")
        plt.title("neg_1_loss")
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(epoch_datas, neg_2_loss, label="neg_2_loss")
        plt.title("neg_2_loss")
        plt.legend()

        plt.suptitle("{}_loss".format(config.image_name))
        plt.savefig('../'+ config.file +'_loss/{}_nice.png'.format(config.image_name), dpi=200)
        # plt.show()

    def prepare_data_for_d(self, index, flag):
        if(flag):
            nodes = self.node_list[index * config.batch_size: (index + 1) * config.batch_size]
            node_pos, relation_pos, node_neighbor_pos, neg_node, neg_relation, neg_node_neighbor = random_walk(self.c_adj_g,
                                                                                                           nodes,
                                                                                                           self.m_type,
                                                                                                           self.graph,
                                                                                                           self.t_relation)
            pos_node_ids = node_pos
            pos_relation_ids = relation_pos
            pos_node_neighbor_ids = node_neighbor_pos

            neg_node_ids_1 = copy.copy(neg_node)
            neg_relation_ids_1 = copy.copy(neg_relation)
            neg_node_neighbor_ids_1 = copy.copy(neg_node_neighbor)

            neg_node_ids_2 = copy.copy(node_pos)
            neg_relation_ids_2 = copy.copy(relation_pos)
            neg_node_neighbor_ids_2 = copy.copy(node_neighbor_pos)

        else:

            pos_node_ids = []
            pos_relation_ids = []
            pos_node_neighbor_ids = []

            #real node and wrong relation
            neg_node_ids_1 = []
            neg_relation_ids_1 = []
            neg_node_neighbor_ids_1 = []

            #fake node and true relation
            neg_node_ids_2 = []
            neg_relation_ids_2 = []
            neg_node_neighbor_ids_2 = []
            node_fake_neighbor_embedding = None



            for node_id in self.node_list[index * config.batch_size : (index + 1) * config.batch_size]:
                for i in range(config.n_sample):

                    # sample real node and true relation node_id, relation_id， node_neighbor_id 确定
                    relations = list(self.graph[node_id].keys())
                    relation_id = relations[np.random.randint(0, len(relations))]
                    neighbors = self.graph[node_id][relation_id]
                    node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]

                    pos_node_ids.append(node_id)
                    pos_relation_ids.append(relation_id)
                    pos_node_neighbor_ids.append(node_neighbor_id)

                    #sample real node and wrong relation
                    neg_node_ids_1.append(node_id)
                    neg_node_neighbor_ids_1.append(node_neighbor_id)
                    neg_relation_id_1 = np.random.randint(0, self.n_relation)
                    while neg_relation_id_1 == relation_id:
                        neg_relation_id_1 = np.random.randint(0, self.n_relation)
                    neg_relation_ids_1.append(neg_relation_id_1)

                    #sample fake node and true relation
                    neg_node_ids_2.append(node_id)
                    neg_relation_ids_2.append(relation_id)
                    neg_node_neighbor_ids_2.append(node_neighbor_id)

        # generate fake node
        noise_embedding = np.random.normal(0.0, config.sig, (len(neg_node_ids_2), config.n_emb)) # 高斯分布方差 sig ^ 2 * I

        node_fake_neighbor_embedding, node_fake_neighbor_embedding_j = self.sess.run([self.generator.node_neighbor_embedding, self.generator.node_neighbor_embedding_j],
                                                     feed_dict = {self.generator.node_id : np.array(neg_node_ids_2),
                                                                  self.generator.relation_id : np.array(neg_relation_ids_2),
                                                                  self.generator.noise_embedding : np.array(noise_embedding),
                                                                  self.generator.node_neighbor_id : np.array(neg_node_neighbor_ids_2)})

        return pos_node_ids, pos_relation_ids, pos_node_neighbor_ids, neg_node_ids_1, neg_relation_ids_1, \
               neg_node_neighbor_ids_1, neg_node_ids_2, neg_relation_ids_2, neg_node_neighbor_ids_2, node_fake_neighbor_embedding, node_fake_neighbor_embedding_j

    def prepare_data_for_g(self, index, flag):
        if(flag):
            nodes = self.node_list[index * config.batch_size: (index + 1) * config.batch_size]

            node_pos, relation_pos, node_neighbor_pos, neg_node, neg_relation, neg_node_neighbor = random_walk(
                self.c_adj_g, nodes, self.m_type, self.graph, self.t_relation)
            node_ids = copy.copy(node_pos)
            relation_ids = copy.copy(relation_pos)
            node_neighbor_ids = copy.copy(node_neighbor_pos)
        else:
            node_ids = []
            relation_ids = []
            node_neighbor_ids = []



            for node_id in self.node_list[index * config.batch_size : (index + 1) * config.batch_size]:
                for i in range(config.n_sample):
                    relations = list(self.graph[node_id].keys())
                    relation_id = relations[np.random.randint(0, len(relations))]
                    neighbors = self.graph[node_id][relation_id]
                    node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]

                    node_ids.append(node_id)
                    relation_ids.append(relation_id)
                    node_neighbor_ids.append(node_neighbor_id)


        noise_embedding = np.random.normal(0.0, config.sig, (len(node_ids), config.n_emb))
        # print(noise_embedding)

        dis_node_embedding, dis_relation_embedding, dis_node_neighbor_embedding = self.sess.run([self.discriminator.pos_node_embedding, self.discriminator.pos_relation_embedding, self.discriminator.pos_node_neighbor_embedding],
                                                                    feed_dict = {self.discriminator.pos_node_id : np.array(node_ids),
                                                                                 self.discriminator.pos_relation_id : np.array(relation_ids),
                                                                                 self.discriminator.pos_node_neighbor_id : np.array(node_neighbor_ids)})

        return node_ids, relation_ids, node_neighbor_ids, noise_embedding, dis_node_embedding, dis_relation_embedding, dis_node_neighbor_embedding


# ----------------aminer----------------
    def evaluate_aminer_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.aminer_evaluation.evaluate_aminer_cluster(embedding_matrix)
            scores.append(score)

        return scores  # NMI

    def evaluate_aminer_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.aminer_evaluation.evaluate_aminer_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_aminer_link_prediction(self):
        modes = [self.generator, self.discriminator]
        aucs = []
        f1s = []
        accs = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)

            auc, f1, acc = self.aminer_evaluation.aminer_link_prediction(embedding_matrix)
            aucs.append(auc)
            f1s.append(f1)
            accs.append(acc)

            print('auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc))
        return aucs, accs, f1s

# ----------------------acm------------------------
    def evaluate_acm_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.acm_evaluation.evaluate_acm_cluster(embedding_matrix)
            scores.append(score)

        return scores  # NMI

    def evaluate_acm_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.acm_evaluation.evaluate_acm_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_acm_link_prediction(self):
        modes = [self.generator, self.discriminator]
        aucs = []
        f1s = []
        accs = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)

            auc, f1, acc = self.acm_evaluation.evaluation_link_prediction(embedding_matrix)
            aucs.append(auc)
            f1s.append(f1)
            accs.append(acc)

            print('auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc))
        return aucs, accs, f1s

    # ------------------


    def evaluate_author_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.dblp_evaluation.evaluate_author_cluster(embedding_matrix)
            scores.append(score)

        return scores # NMI

    def evaluate_author_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.dblp_evaluation.evaluate_author_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_paper_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.aminer_evaluation.evaluate_paper_cluster(embedding_matrix)
            scores.append(score)

        return scores

    def evaluate_paper_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.aminer_evaluation.evaluate_paper_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_business_cluster(self):
        modes = [self.generator, self.discriminator]
        scores = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            score = self.yelp_evaluation.evaluate_business_cluster(embedding_matrix)
            scores.append(score)

        return scores

    def evaluate_business_classification(self):
        modes = [self.generator, self.discriminator]
        micro_f1s = []
        macro_f1s = []
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            micro_f1, macro_f1 = self.yelp_evaluation.evaluate_business_classification(embedding_matrix)
            micro_f1s.append(micro_f1)
            macro_f1s.append(macro_f1)
        return micro_f1s, macro_f1s

    def evaluate_yelp_link_prediction(self):
        modes = [self.generator, self.discriminator]
        aucs = []
        f1s = []
        accs = []
        sample_8, sample_2 = utils.get_neg_sample(self.neg_g, config.node_size, self.size)

        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)

            #score = self.yelp_evaluation.evaluate_business_cluster(embedding_matrix)
            #print() '%d nmi = %.4f' % (i, score)

            auc, f1, acc = self.yelp_evaluation.evaluation_link_prediction(embedding_matrix)
            aucs.append(auc)
            f1s.append(f1)
            accs.append(acc)

            print('auc = %.4f f1 = %.4f acc = %.4f' % (auc, f1, acc))
        return aucs, accs, f1s

    def evaluate_dblp_link_prediction(self):
        modes = [self.generator, self.discriminator]
        aucs = []
        f1s = []
        accs = []

        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            #relation_matrix = self.sess.run(modes[i].relation_embedding_matrix)

            auc, f1, acc = self.dblp_evaluation.evaluation_link_prediction(embedding_matrix)

            aucs.append(auc)
            f1s.append(f1)
            accs.append(acc)

        return aucs, accs, f1s



    def write_embeddings_to_file(self, epoch):
        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = self.sess.run(modes[i].node_embedding_matrix)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + ' ' + ' '.join([str(x) for x in emb[1:]]) + '\n' for emb in embedding_list]

            with open(config.emb_filenames[i], 'w') as f:
                lines = [str(self.n_node) + ' ' + str(config.n_emb) + '\n'] + embedding_str
                f.writelines(lines)

    def advance_train(self, k1=0, k2=config.alaph):
        self.train(k1, 1, k2, "", 0)
        # self.train(k1, 0, "origin")
        # self.train(k2, 1, "advance")

if __name__ == '__main__':
        model = Model()
        model.advance_train()
