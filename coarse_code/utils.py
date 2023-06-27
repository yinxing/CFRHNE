import numpy as np
import config
from collections import defaultdict

t_size = config.t_size

def read_graph():
    #dblp                          ----------1----------
    # p -> a : 0  p: paper, t : term, a : author
    # a -> p : 1
    # p -> c : 2
    # c -> p : 3
    # p -> t : 4
    # t -> p : 5
    # p : 0, a : 1, c : 2, t : 3
    # self.t_relation = np.array([[-1, 0, 2, 4], [1, -1, -1, -1], [3, -1, -1, -1], [5, -1, -1, -1]])

    #yelp
    # 0 -> 1: 0
    # 0 -> 2: 2
    # 0 -> 3: 4
    # 0 -> 4: 6
    # 1 -> 0: 1
    # 2 -> 0: 3
    # 3 -> 0: 5
    # 7 -> 0: 7
    # self.t_relation = np.array([[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])
    net = config.network
    dataset = config.dataset
    graph_filename = "../" + net + "/" + dataset + "/links.dat"

    relations = set()
    nodes = set()
    graph = {}
    g_type = np.zeros([t_size, t_size])
    m_type = {}
    adj_g = {}
    c_adj_g = {}
    neg_g = np.zeros((config.node_size, config.node_size),dtype=bool)
    cnt = 0

    #map_type() 用于区分两节点类型的

    with open(graph_filename) as infile:
        for line in infile.readlines():
            cnt += 1
            source_node, target_node, relation = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)
            relation = int(relation)
            neg_g[source_node][target_node] = True

            if(target_node not in adj_g):                         #收入边的节点
                adj_g[target_node] = []
            if(source_node not in c_adj_g):
                c_adj_g[source_node] = []
            adj_g[target_node].append(source_node)
            c_adj_g[source_node].append(target_node)
            #dblp            ------------2-----------
            # if(relation == 0):
            #     g_type[0][1] += 1
            #     m_type[source_node] = 0
            # elif(relation == 1):
            #     g_type[1][0] += 1
            #     m_type[source_node] = 1
            # elif(relation == 2):
            #     g_type[0][2] += 1
            #     m_type[source_node] = 0
            # elif(relation == 3):
            #     g_type[2][0] += 1
            #     m_type[source_node] = 2
            # elif(relation == 4):
            #     g_type[0][3] += 1
            #     m_type[source_node] = 0
            # elif(relation == 5):
            #     g_type[3][0] += 1
            #     m_type[source_node] = 3
            # else:
            #     pass

            # yelp
            # 0 -> 1: 0
            # 0 -> 2: 2
            # 0 -> 3: 4
            # 0 -> 4: 6
            # 1 -> 0: 1
            # 2 -> 0: 3
            # 3 -> 0: 5
            # 4 -> 0: 7

            if(relation == 0):
                g_type[0][1] += 1
                m_type[source_node] = 0
            elif(relation == 1):
                g_type[1][0] += 1
                m_type[source_node] = 1
            elif(relation == 2):
                g_type[0][2] += 1
                m_type[source_node] = 0
            elif(relation == 3):
                g_type[2][0] += 1
                m_type[source_node] = 2
            elif(relation == 4):
                g_type[0][3] += 1
                m_type[source_node] = 0
            elif(relation == 5):
                g_type[3][0] += 1
                m_type[source_node] = 3
            elif(relation == 6):
                g_type[0][4] += 1
                m_type[source_node] = 0
            elif (relation == 7):
                g_type[4][0] += 1
                m_type[source_node] = 4
            else:
                pass

            nodes.add(source_node)
            nodes.add(target_node)
            relations.add(relation)

            if source_node not in graph:
                graph[source_node] = {}

            if relation not in graph[source_node]:
                graph[source_node][relation] = []

            graph[source_node][relation].append(target_node)

    norm_g_type = Normalized(g_type)
    #print relations
    n_node = len(nodes)
    return n_node, len(relations), graph, g_type, norm_g_type, m_type, cnt, neg_g, adj_g, c_adj_g

def reverse_type(M):
    type = {}
    for i in range(config.t_size):
        type[i] = []
    for i in range(config.node_size):
        type[M[i]].append(i)
    return type



def softmax(x):
    """ softmax function """

    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x

def Normalized_weight(weight):
    w = {}
    for i in list(weight.keys()):
        w[i] = softmax(weight[i])
    return w

def all_Normalized(g_type):
    sum = np.sum(g_type)
    norm_g_type = np.zeros(g_type.shape)
    for i in range(g_type.shape[0]):
        for j in range(g_type.shape[1]):
            norm_g_type[i][j] = float(g_type[i][j]) / sum
    return norm_g_type

def Normalized(g_type):

    sum = np.sum(g_type,axis=1)
    norm_g_type = np.zeros(g_type.shape)
    for i in range(g_type.shape[0]):
        for j in range(g_type.shape[1]):
            norm_g_type[i][j] = float(g_type[i][j]) / sum[i]
    return norm_g_type

def generate(g):
    size = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if (g[i][j] != 0):
                size += 1

    edge_index = np.zeros((2, size), dtype=float)
    edge_w = np.zeros(size,dtype=float)

    k = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if(g[i][j] != 0):
                edge_index[0][k] = i
                edge_index[1][k] = j
                edge_w[k] = g[i][j]
                k += 1
    return edge_index, edge_w

def ploy_init(adj, M, g_type):
    a = {}
    for i in range(config.node_size):
        t = []
        ti = M[i]
        for j in adj[i]:
            tj = M[j]
            w = g_type[ti][tj]
            t.append(w)
        a[i] = t
    return a

def generate_n_edge(adj, w, size):
    edge_index = np.zeros((2, size), dtype=float)
    edge_w = np.zeros(size, dtype=float)

    k = 0
    for source in range(1):
        for index, t_node in enumerate(adj[source]):
            edge_index[0][k] = t_node
            edge_index[1][k] = source
            edge_w[k] = w[source][index]
            k += 1

    return edge_index, edge_w

def map_type():  #将节点编号映射成节点类型
    type_path = []
    m_type = {}

    #加入节点类型的文件

    for item in type_path:
        with open(item) as infile:
            for line in infile.readlines():
                source_node, type = line.strip().split(' ')
                m_type[source_node] = type
    return m_type

def get_relation(node_list, n_relation, g):
    r_graph = np.zeros((n_relation, n_relation))
    for source_node in node_list:
        relations = list(g[source_node].keys())
        for r in relations:
            target_nodes = list(g[source_node][r])
            for node in target_nodes:
                target_realtions = list(g[node].keys())
                for t_r in target_realtions:
                    cnt = len(list(g[node][t_r]))
                    r_graph[r][t_r] += cnt
    return r_graph

def get_neg_sample(g, total, cnt):
    sample_8 = []
    sample_2 = []
    for i in range(cnt):
        a = np.random.randint(total)
        b = np.random.randint(total)
        while(g[a][b]):
            a = np.random.randint(total)
            b = np.random.randint(total)
        c = np.random.rand()
        if(c <= 0.8):
            sample_8.append([a, b, 0])
        else:
            sample_2.append([a, b, 0])
    return sample_8, sample_2

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def read_embeddings(filename, n_node, n_embed):

    embedding_matrix = np.random.rand(n_node, n_embed)
    i = -1
    with open(filename) as infile:
        for line in infile.readlines()[1:]:
            i += 1
            emd = line.strip().split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    return embedding_matrix
def create_n_edge(node_neighbors, nodes, weight, size):
    type_edge_source = defaultdict(list)
    type_edge_target = defaultdict(list)

    for node in nodes:
        for type in range(config.t_size):
            for next_node in node_neighbors[node][type]:
                type_edge_source[type].append(next_node)
                type_edge_target[type].append(node)

    type_edge_index = {}
    for type in range(config.t_size):
        assert len(type_edge_source[type]) == len(type_edge_target[type])
        size = len(type_edge_target[type])
        type_edge_index[type] = np.zeros((2, size), dtype=int)
        for j in range(size):
            type_edge_index[type][0][j] = type_edge_source[type][j]
            type_edge_index[type][1][j] = type_edge_target[type][j]
    return type_edge_index


def create_all_edge(g, nodes):
    edge_index_source = []
    edge_index_target = []

    for node in range(nodes):
       for node_neighbor in g[node]:
            edge_index_source.append(node_neighbor)
            edge_index_target.append(node)
    return np.array([edge_index_source, edge_index_target])

if __name__ == '__main__':
    n_node, n_relation, graph  = read_graph()

    #embedding_matrix = read_embeddings('../data/dblp/rel_embeddings.txt', 6, 64)
    print(graph[1][1])
