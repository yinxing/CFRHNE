# import numpy as np
# import config
# from collections import defaultdict
# from collections import Counter
# import random
# import utils
# from tqdm import tqdm
#
# def random_walk(adj_g, nodes, m_type, graph, t_relation):
#     node_ids = []
#     relation_ids = []
#     node_neighbor_ids = []
#
#     neg_node_ids = []
#     neg_relation_ids = []
#     neg_node_neighbor_ids = []
#
#
#     WALK_LEN = config.walk_len
#     WALK_NUM = config.walk_num
#
#     for node in nodes:
#         for i in range(WALK_NUM):
#             relation_id = random.choice(list(graph[node].keys()))
#             cur_node = random.choice(graph[node][relation_id])
#
#             node_ids.append(node)
#             relation_ids.append(relation_id)
#             node_neighbor_ids.append(cur_node)
#
#             for j in range(WALK_LEN):
#                 next_node = random.choice(list(adj_g[cur_node]))
#                 if(next_node == node): continue
#                 else:
#                     if(relation_id == t_relation[m_type[node]][m_type[next_node]]):
#                         node_ids.append(node)
#                         relation_ids.append(relation_id)
#                         node_neighbor_ids.append(next_node)
#                     else:
#                         neg_node_ids.append(node)
#                         neg_relation_ids.append(relation_id)
#                         neg_node_neighbor_ids.append(next_node)
#
#                     cur_node = next_node
#
#     return node_ids, relation_ids, node_neighbor_ids, neg_node_ids, neg_relation_ids, neg_node_neighbor_ids
#
# if __name__ == "__main__":
#     nodes = np.arange(2)
#     n_node, n_relation, graph, g_type, norm_gt, m_type, size, neg_g, r_adj_g, c_adj_g = utils.read_graph(config.graph_filename)
#     t_relation = np.array(
#         [[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])
#     node_ids, relation_ids, node_neighbor_ids, neg_node_ids, neg_relation_ids, neg_node_neighbor_ids = random_walk(c_adj_g, nodes, m_type, graph, t_relation)
#

#
# import numpy as np
# import config
# from collections import defaultdict
# from collections import Counter
# import random
# import utils
# from tqdm import tqdm
#
# def random_walk(adj_g, nodes, m_type, graph, t_relation):
#     node_ids = []
#     relation_ids = []
#     node_neighbor_ids = []
#
#     neg_node_ids = []
#     neg_relation_ids = []
#     neg_node_neighbor_ids = []
#
#
#     WALK_LEN = config.walk_len
#     WALK_NUM = config.walk_num
#
#     for node in nodes:
#         for i in range(WALK_NUM):
#             relation_id = random.choice(list(graph[node].keys()))
#             cur_node = random.choice(graph[node][relation_id])
#             cnt = 0
#
#             # node_ids.append(node)
#             # relation_ids.append(relation_id)
#             # node_neighbor_ids.append(cur_node)
#             # cnt += 1
#
#             visit_node = []
#             visit_node.append(node)
#             for j in range(WALK_LEN):
#                 visit_node.append(cur_node)
#                 next_node = random.choice(list(adj_g[cur_node]))
#                 if(next_node == node): continue
#                 else:
#                     if(relation_id == t_relation[m_type[node]][m_type[next_node]]):
#                         node_ids.append(node)
#                         relation_ids.append(relation_id)
#                         node_neighbor_ids.append(next_node)
#                         cnt += 1
#                 cur_node = next_node
#
#             for j in range(cnt):
#                 neg_node_id = np.random.randint(config.node_size)
#                 # print(neg_node_id)
#                 while(neg_node_id in visit_node or relation_id == t_relation[m_type[node]][m_type[neg_node_id]]):
#                     neg_node_id = np.random.randint(config.node_size)
#                 neg_node_ids.append(node)
#                 neg_relation_ids.append(relation_id)
#                 neg_node_neighbor_ids.append(neg_node_id)
#
#     return node_ids, relation_ids, node_neighbor_ids, neg_node_ids, neg_relation_ids, neg_node_neighbor_ids
#
# if __name__ == "__main__":
#     nodes = np.arange(2)
#     n_node, n_relation, graph, g_type, norm_gt, m_type, size, neg_g, r_adj_g, c_adj_g = utils.read_graph(config.graph_filename)
#     t_relation = np.array(
#         [[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])
#     node_ids, relation_ids, node_neighbor_ids, neg_node_ids, neg_relation_ids, neg_node_neighbor_ids = random_walk(c_adj_g, nodes, m_type, graph, t_relation)
#     print(node_ids)
#     print(relation_ids)
#     print(node_neighbor_ids)
#     print(neg_node_neighbor_ids)
#     print(len(node_ids))
#     print(len(node_neighbor_ids))
#     print(len(neg_node_neighbor_ids))
#


import numpy as np
import config
from collections import defaultdict
from collections import Counter
import random
import utils
from tqdm import tqdm

def random_walk(adj_g, nodes, m_type, graph, t_relation):
    node_ids = []
    relation_ids = []
    node_neighbor_ids = []

    neg_node_ids = []
    neg_relation_ids = []
    neg_node_neighbor_ids = []


    WALK_LEN = config.walk_len
    WALK_NUM = config.walk_num

    for node in nodes:
        for i in range(WALK_NUM):
            relation_id = random.choice(list(graph[node].keys()))
            cur_node = random.choice(graph[node][relation_id])
            cnt = 0

            # node_ids.append(node)
            # relation_ids.append(relation_id)
            # node_neighbor_ids.append(cur_node)
            # cnt += 1

            visit_node = []
            visit_node.append(node)
            for j in range(WALK_LEN):
                visit_node.append(cur_node)
                next_node = random.choice(list(adj_g[cur_node]))
                if(next_node == node): continue
                else:
                    if(relation_id == t_relation[m_type[node]][m_type[next_node]]):
                        node_ids.append(node)
                        relation_ids.append(relation_id)
                        node_neighbor_ids.append(next_node)
                        cnt += 1
                cur_node = next_node

            for j in range(cnt):
                neg_node_id = np.random.randint(config.node_size)
                # print(neg_node_id)
                while(neg_node_id in visit_node or relation_id == t_relation[m_type[node]][m_type[neg_node_id]]):
                    neg_node_id = np.random.randint(config.node_size)
                neg_node_ids.append(node)
                neg_relation_ids.append(relation_id)
                neg_node_neighbor_ids.append(neg_node_id)

    return node_ids, relation_ids, node_neighbor_ids, neg_node_ids, neg_relation_ids, neg_node_neighbor_ids

if __name__ == "__main__":
    nodes = np.arange(2)
    n_node, n_relation, graph, g_type, norm_gt, m_type, size, neg_g, r_adj_g, c_adj_g = utils.read_graph(config.graph_filename)
    t_relation = np.array(
        [[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])
    node_ids, relation_ids, node_neighbor_ids, neg_node_ids, neg_relation_ids, neg_node_neighbor_ids = random_walk(c_adj_g, nodes, m_type, graph, t_relation)
    print(node_ids)
    print(relation_ids)
    print(node_neighbor_ids)
    print(neg_node_neighbor_ids)
    print(len(node_ids))
    print(len(node_neighbor_ids))
    print(len(neg_node_neighbor_ids))
