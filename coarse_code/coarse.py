import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.data import Data
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import config
import global_init as gol
import utils
import yelp_evaluation
import matplotlib.pyplot as plt
import random_walk_restart
import torch.nn as nn
import random
import evaluation
import time

t_dimension = config.n_emb
t_size = config.t_size
alaph = config.alaph
cnt = 0

seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gol.init_val()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



class GCN(torch.nn.Module):
    def __init__(self, num_node_features, dropout, m_type):
        super(GCN, self).__init__()

        self.type = []

        for i in range(config.t_size * 2):
            gcn = GCNConv(num_node_features, num_node_features)
            setattr(self, 'type_{}_{}'.format(i // 2, i % 2), gcn)
            self.type.append(gcn)



        # 注意力机制
        self.W = nn.Parameter(torch.empty(size=(num_node_features, num_node_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*num_node_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.A = nn.Parameter(torch.empty(size=(2, 1)))
        nn.init.xavier_uniform_(self.A.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.y = torch.zeros((config.node_size, config.n_emb))


        self.M = m_type
        self.out_features = num_node_features



# —————————————修改weight———————————————
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        Y = []
        for i in range(config.t_size):
            y = self.type[i * 2](x[i, :, :], torch.from_numpy(edge_index[i]).type(torch.long).to(device))
            y = self.leakyrelu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = self.type[i * 2 + 1](y, torch.from_numpy(edge_index[i]).type(torch.long).to(device))
            y = self.leakyrelu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            Y.append(y)

        return Y


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):  # , ,丢失率 ,斜率，是否使用elu
        super(GraphAttentionLayer, self).__init__()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.gat = GATConv(in_features, out_features).to(device)

    def forward(self, x, edge_index):
        y = self.gat(x, edge_index)
        y = self.leakyrelu(y)
        y = F.dropout(y, p=self.dropout, training=self.training)

        return y

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, m_type, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat * config.t_size, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, type_emb, edge):


        x = torch.cat([att(torch.FloatTensor(np.concatenate(type_emb[i], axis=1)).to(device), edge) for i, att in enumerate(self.attentions)], dim=1)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, edge)
        x = F.tanh(x)
        return x

    # --------------method----------------
def train(model, type_emb, graph_edge_index, pos_st, pos_ed, neg_st, neg_ed):



    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    model.train()

    for index in range(config.t_time):
        out = model(type_emb, graph_edge_index)

        optimizer.zero_grad()

        pos_emb_st = torch.index_select(out, 0, torch.tensor(pos_st).to(device))
        pos_emb_ed = torch.index_select(out, 0, torch.tensor(pos_ed).to(device))

        neg_emb_st = torch.index_select(out, 0, torch.tensor(neg_st).to(device))
        neg_emb_ed = torch.index_select(out, 0, torch.tensor(neg_ed).to(device))

        pos_sorce = get_scores(pos_emb_st, pos_emb_ed)
        neg_sorce = get_scores(neg_emb_st, neg_emb_ed)



        loss_1 = F.binary_cross_entropy(torch.sigmoid(pos_sorce), torch.ones(pos_sorce.shape).to(device))
        loss_2 = F.binary_cross_entropy(torch.sigmoid(neg_sorce), torch.zeros(neg_sorce.shape).to(device))
        loss = loss_2 + loss_1
        print(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        loss.backward()
        optimizer.step()


def first_train_enter(model, data, pos_st, pos_ed, neg_st, neg_ed):



    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    model.train()

    for index in range(config.t_time):
        Y = model(data)

        optimizer.zero_grad()

        loss = 0
        for out in Y:

            pos_emb_st = torch.index_select(out, 0, torch.tensor(pos_st).to(device))
            pos_emb_ed = torch.index_select(out, 0, torch.tensor(pos_ed).to(device))

            neg_emb_st = torch.index_select(out, 0, torch.tensor(neg_st).to(device))
            neg_emb_ed = torch.index_select(out, 0, torch.tensor(neg_ed).to(device))

            pos_sorce = get_scores(pos_emb_st, pos_emb_ed)
            neg_sorce = get_scores(neg_emb_st, neg_emb_ed)



            loss_1 = F.binary_cross_entropy(torch.sigmoid(pos_sorce), torch.ones(pos_sorce.shape).to(device))
            loss_2 = F.binary_cross_entropy(torch.sigmoid(neg_sorce), torch.zeros(neg_sorce.shape).to(device))

            loss = loss + loss_2 + loss_1
            # print(loss)
            #
            # sorces = torch.mm(out, out.t())
            # loss = F.binary_cross_entropy(torch.sigmoid(sorces), torch.FloatTensor(adj_matrix))

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        loss.backward()
        optimizer.step()

def second_train_enter(model, type_emb, pos_st, pos_ed, neg_st, neg_ed, device):

    x0, x1, x2, x3, x4 = type_emb
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    model.train()



    for index in range(config.t_time):
        out = model(x0, x1, x2, x3, x4)

        optimizer.zero_grad()

        pos_emb_st = torch.index_select(out, 0, torch.tensor(pos_st).to(device))
        pos_emb_ed = torch.index_select(out, 0, torch.tensor(pos_ed).to(device))

        neg_emb_st = torch.index_select(out, 0, torch.tensor(neg_st).to(device))
        neg_emb_ed = torch.index_select(out, 0, torch.tensor(neg_ed).to(device))

        pos_sorce = get_scores(pos_emb_st, pos_emb_ed)
        neg_sorce = get_scores(neg_emb_st, neg_emb_ed)



        loss_1 = F.binary_cross_entropy(torch.sigmoid(pos_sorce), torch.ones(pos_sorce.shape))
        loss_2 = F.binary_cross_entropy(torch.sigmoid(neg_sorce), torch.zeros(neg_sorce.shape))

        loss = loss_2 + loss_1
        #
        # sorces = torch.mm(out, out.t())
        # loss = F.binary_cross_entropy(torch.sigmoid(sorces), torch.FloatTensor(adj_matrix))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        loss.backward()
        optimizer.step()

def get_scores(a, b):
    return torch.multiply(a, b).sum(dim=1)

def first_train(model, x, edge_index):
    pos_st = np.array(gol.get_value('pos_st'), dtype=int)
    pos_ed = np.array(gol.get_value('pos_ed'), dtype=int)
    neg_st = np.array(gol.get_value('neg_st'), dtype=int)
    neg_ed = np.array(gol.get_value('neg_ed'), dtype=int)

    # train_mask = torch.from_numpy(np.ones(config.t_size, dtype=bool))
    #
    x = torch.from_numpy(x).type(torch.float32).to(device)
    data = Data(x=x, edge_index=edge_index)
    # num_node_features = config.n_emb
    # _device = 'cpu'
    first_train_enter(model, data, pos_st, pos_ed, neg_st, neg_ed)
    model.eval()
    Y = model(data)
    t_feature = []
    for out in Y:
        t_feature.append(out.cpu().detach().numpy())
    return t_feature

def second_train(model, type_emb):
    pos_st = np.array(gol.get_value('pos_st'), dtype=int)
    pos_ed = np.array(gol.get_value('pos_ed'), dtype=int)
    neg_st = np.array(gol.get_value('neg_st'), dtype=int)
    neg_ed = np.array(gol.get_value('neg_ed'), dtype=int)

    train(model, type_emb, graph_edge_index, pos_st, pos_ed, neg_st, neg_ed)
    with torch.no_grad():
        model.eval()
        t_feature = model(type_emb, graph_edge_index)
    return t_feature.cpu().detach().numpy()


def global_sample(nodes, m_type, type_node, g):
    pos_st = []
    pos_ed = []
    neg_st = []
    neg_ed = []

    for node_id in nodes:
        rs = list(g[node_id].keys())
        for r in rs:
            neighbors = g[node_id][r]
            for i in range(config.pos_size):
                node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]
                pos_st.append(node_id)
                pos_ed.append(node_neighbor_id)

    for node_id in nodes:
        rs = list(g[node_id].keys())
        for r in rs:
            type = m_type[g[node_id][r][0]]
            neg_neighbors = type_node[type]
            for i in range(config.neg_size):

                node_neighbor_id = neg_neighbors[np.random.randint(0, len(neg_neighbors))]
                while(node_neighbor_id in g[node_id][r]):
                    node_neighbor_id = neg_neighbors[np.random.randint(0, len(neg_neighbors))]

                neg_st.append(node_id)
                neg_ed.append(node_neighbor_id)

    return pos_st, pos_ed, neg_st, neg_ed

def evaluate_business_cluster(Y, emb):
    score = Y.evaluate_business_cluster(emb)

    print("___________________cluster_______________________")
    print(' NMI score = %.4f' % (score))
    return score

def evaluate_business_classification(Y, emb):

    micro_f1, macro_f1 = Y.evaluate_business_classification(emb)
    print("_________________clasfiction______________________")
    print(' micro_f1 = %.4f  micro_f1 = %.4f' % (micro_f1, macro_f1))
    return micro_f1, macro_f1

def evaluate_yelp_link_prediction(Y, neg_g, emb, size):

    sample_8, sample_2 = utils.get_neg_sample(neg_g, config.node_size, size)

    auc, f1, acc = Y.evaluation_link_prediction(emb, sample_8, sample_2)
    print("_________________link_prediction______________________")
    print(' auc = %.4f  f1 = %.4f acc = %.4f' % (auc, f1, acc))
    return f1, acc, auc



def draw(epoch_datas, nmis, f1s, accs, aucs, mi_f1s, ma_f1s):
    plt.subplot(2, 2, 1)
    # my_y_ticks = np.arange(0.2, 0.4, 0.005)
    # my_x_ticks = np.arange(0, config.n_epoch, 1)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    plt.plot(epoch_datas, nmis, label="nmi")
    plt.title("NMI")
    plt.legend()

    # plot 2:

    plt.subplot(2, 2, 2)
    # my_y_ticks = np.arange(0.68, 0.83, 0.005)
    # my_x_ticks = np.arange(0, config.n_epoch, 1)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    plt.plot(epoch_datas, f1s, label="f1")

    plt.plot(epoch_datas, accs, label="acc")

    plt.plot(epoch_datas, aucs, label="auc")

    plt.title("link_prediction")
    plt.legend()

    # plot 3:

    plt.subplot(2, 2, 3)
    # my_y_ticks = np.arange(0.68, 0.79, 0.005)
    # my_x_ticks = np.arange(0, config.n_epoch, 1)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    plt.plot(epoch_datas, mi_f1s, label="micro_f1")
    plt.plot(epoch_datas, ma_f1s, label="macro_f1")
    plt.title("classification")
    plt.legend()

    plt.suptitle("{}".format(config.image_name))
    plt.savefig('../t_dimension/{}.png'.format(config.image_name), dpi=200)
    plt.show()

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
#
# def pre_deal(t):
#     h = np.loadtxt('H')

if __name__ == '__main__':

    dataset = config.dataset
    n_node, n_relation, graph, g_type, norm_gt, m_type, size, neg_g, adj_g, c_adj_g = utils.read_graph()
    # print(n_node)
    # exit(0)
    type_node = utils.reverse_type(m_type)
    t_relation = np.array(
        [[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])

    graph_edge_index = utils.create_all_edge(c_adj_g, config.node_size)
    graph_edge_index = torch.from_numpy(graph_edge_index).type(torch.long).to(device)


    t = np.random.normal(size=(config.t_size, config.node_size, config.n_emb))
    # t = pre_deal(t)
    # print(adj_matrix)

    # print(len(type_node[0]))
    # print(len(type_node[1]))
    # print(len(type_node[2]))
    # exit(0)
    # print(t)
    # print("-------------------------------------------------")
    epoch_datas = []
    nmis = []
    f1s = []
    accs = []
    aucs = []
    mi_f1s = []
    ma_f1s = []

    # type_node_indexs = []
    # for _ in range(8):
        # type_node_indexs.append(type_node_index)
    node_list = np.arange(config.node_size)
    num_node_features = t_dimension

    y = norm_gt + np.eye(norm_gt.shape[0])

    norm_sgt = utils.softmax(y)
    #
    max_nmi = 0.
    type_embs = []
    for i in range(config.n_heads):
        model = GCN(num_node_features, 0.2, m_type).to(device)
        for i in range(20):
            np.random.shuffle(node_list)
            print('______________epoch %d ________________' % (i))
            for index in range(len(node_list) // config.batch_size):
                print('______________batch %d ________________' % (index))
                nodes = node_list[index * config.batch_size: (index + 1) * config.batch_size]
                pos_st, pos_ed, neg_st, neg_ed = global_sample(nodes, m_type, type_node, graph)
                gol.set_value('pos_st', pos_st)
                gol.set_value('neg_st', neg_st)
                gol.set_value('pos_ed', pos_ed)
                gol.set_value('neg_ed', neg_ed)

                # node_neighbors = random_walk_restart.random_walk_restart(norm_gt, c_adj_g, nodes, m_type, t_relation, graph)
                node_neighbors = random_walk_restart.random_walk_restart(c_adj_g, nodes, m_type)
                type_node_index = utils.create_n_edge(node_neighbors, nodes, [], size)
                type_emb = first_train(model, t, type_node_index)
            print(type_emb)
            # exit(0)
        type_embs.append(type_emb)



    path = '../new_emb/' + dataset + '/gcn_20_seed_0.npz'
    np.savez(path, type_emb = type_embs)
    #
    # exit(0)

    t_data = np.load(path)
    type_embs = t_data['type_emb']

    model = GAT(num_node_features, num_node_features * 2, num_node_features, 0.2, 0.2, m_type, config.n_heads).to(device)

    for i in range(15):
        np.random.shuffle(node_list)
        print('______________epoch %d ________________' % (i))
        for index in range(len(node_list) // config.batch_size):
            print('______________batch %d ________________' % (index))
            nodes = node_list[index * config.batch_size: (index + 1) * config.batch_size]
            pos_st, pos_ed, neg_st, neg_ed = global_sample(nodes, m_type, type_node, graph)
            gol.set_value('pos_st', pos_st)
            gol.set_value('neg_st', neg_st)
            gol.set_value('pos_ed', pos_ed)
            gol.set_value('neg_ed', neg_ed)
            emb = second_train(model,type_embs)


        # Y = yelp_evaluation.Yelp_evaluation()
        # f1, acc, auc = evaluate_yelp_link_prediction(Y, neg_g, emb, size)
        # nmi = evaluate_business_cluster(Y, emb)
        # mi_f1, ma_f1 = evaluate_business_classification(Y, emb)

        f1, acc, auc, nmi, mi_f1, ma_f1 = evaluation.all_evaluation(emb)
        # if(nmi > max_nmi):
        #     print('nmi: {}'.format(nmi))
        #     np.savetxt('../data/{}.txt'.format(config.image_name), emb)
        #     max_nmi = nmi
        np.savetxt('../emb/{}_{}.txt'.format(config.image_name, i), emb)

        epoch_datas.append(i)
        nmis.append(nmi)
        f1s.append(f1)
        accs.append(acc)
        aucs.append(auc)
        mi_f1s.append(mi_f1)
        ma_f1s.append(ma_f1)

    draw(epoch_datas, nmis, f1s, accs, aucs, mi_f1s, ma_f1s)






