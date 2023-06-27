
#

#
#
#
import numpy as np
import config
from collections import defaultdict
from collections import Counter
import random
import utils
from tqdm import tqdm

def random_walk_restart(adj_g, nodes, m_type):
    node_neighbors = defaultdict(dict)
    for i in nodes:
        node_neighbors[i] = defaultdict(list)
    WALK_LEN = config.walk_len
    for node in nodes:
        cur_node = node
        for j in range(WALK_LEN):
            if(np.random.random() >= config.restart_ratio):
                cur_node = node
                continue
            else:
                next_node = random.choice(list(adj_g[cur_node]))
                cur_node = next_node
            if(node != cur_node):
                node_neighbors[node][m_type[cur_node]].append(cur_node)
    top_k = config.top_k
    for node in nodes:
        for type in range(config.t_size):
            node_tmp = []
            node_neighbors_tmp = Counter(node_neighbors[node][type])
            top_list = node_neighbors_tmp.most_common(top_k[type])
            for i in range(len(top_list)):
                node_tmp.append(top_list[i][0])
            node_neighbors[node][type] = node_tmp
    return node_neighbors
if __name__ == "__main__":
    nodes = np.arange(10)
    n_node, n_relation, graph, g_type, norm_gt, m_type, size, neg_g, r_adj_g, c_adj_g = utils.read_graph()
    node_neighbors = random_walk_restart(c_adj_g, nodes, m_type)
    for i in range(10):
        print(node_neighbors[i])



#
# from queue import Queue
# import numpy as np
# import config
# from collections import defaultdict
# from collections import Counter
# import random
# import utils
# from tqdm import tqdm
#
# def random_walk_restart(adj_g, graph, nodes, m_type, t_relation):
#     node_neighbors = defaultdict(dict)
#     for i in nodes:
#         node_neighbors[i] = defaultdict(list)
#     WALK_LEN = config.walk_len
#     for node in nodes:
#         # Q = Queue(maxsize = 2)
#         cur_node = node
#         flag = 1
#         for j in range(WALK_LEN):
#             # if(np.random.random() >= config.restart_ratio):
#             #     cur_node = node
#             #     continue
#             # else:
#             if(m_type[cur_node] == 0):
#                 all_T = [1, 2, 3, 4]
#                 all_T.remove(flag)
#                 target_t = random.choice(all_T)
#                 # print(m_type[cur_node])
#                 # print(target_t)
#                 next_node = random.choice(list(graph[cur_node][t_relation[m_type[cur_node]][target_t]]))
#                 flag = target_t
#             else:
#                 next_node = random.choice(list(adj_g[cur_node]))
#             cur_node = next_node
#             if(node != cur_node):
#                 node_neighbors[node][m_type[cur_node]].append(cur_node)
#     top_k = config.top_k
#     for node in nodes:
#         for type in range(config.t_size):
#             node_tmp = []
#             node_neighbors_tmp = Counter(node_neighbors[node][type])
#             top_list = node_neighbors_tmp.most_common(top_k[type])
#             for i in range(len(top_list)):
#                 node_tmp.append(top_list[i][0])
#             node_neighbors[node][type] = node_tmp
#     return node_neighbors
# if __name__ == "__main__":
#     nodes = np.arange(10)
#     n_node, n_relation, graph, g_type, norm_gt, m_type, size, neg_g, r_adj_g, c_adj_g = utils.read_graph()
#     t_relation = np.array(
#         [[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])
#
#     node_neighbors = random_walk_restart(c_adj_g, graph, nodes, m_type, t_relation)
#     for i in range(10):
#         print(node_neighbors[i])


