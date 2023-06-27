import numpy as np
import config
import utils

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

if __name__ == "__main__":
    dataset = config.dataset
    n_node, n_relation, graph, g_type, norm_gt, m_type, size, neg_g, adj_g, c_adj_g = utils.read_graph()
    type_node = utils.reverse_type(m_type)
    t_relation = np.array(
        [[-1, 0, 2, 4, 6], [1, -1, -1, -1, -1], [3, -1, -1, -1, -1], [5, -1, -1, -1, -1], [7, -1, -1, -1, -1]])
    sample_8, sample_2 = get_neg_sample(neg_g, config.node_size, size)
    path_8 = "../data/" + dataset + "/neg_0.8"
    path_2 = "../data/" + dataset + "/neg_0.2"

    np.savetxt(path_8, np.array(sample_8), fmt="%d")
    np.savetxt(path_2, np.array(sample_2), fmt="%d")

