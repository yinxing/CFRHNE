



import random

import numpy as np
from collections import defaultdict
import config
import utils





graph_filename = "../data/aminer/links.dat"

t_relation = np.array([[-1, 0, 2, 4], [1, -1, -1, -1], [3, -1, -1, -1], [5, -1, -1, -1]])  # p a c r

graph = {}
adj = defaultdict(list)

for i in range(config.node_size):
    graph[i] = defaultdict(list)

m_type = {}

with open(graph_filename) as infile:
    for line in infile.readlines():
        source_node, target_node, relation = line.strip().split(' ')
        source_node = int(source_node)
        target_node = int(target_node)
        relation = int(relation)
        graph[source_node][relation].append(target_node)
        adj[source_node].append(target_node)

        if (relation == 0):

            m_type[source_node] = 0
        elif (relation == 1):

            m_type[source_node] = 1
        elif (relation == 2):

            m_type[source_node] = 0
        elif (relation == 3):

            m_type[source_node] = 2
        elif (relation == 4):

            m_type[source_node] = 0
        elif (relation == 5):

            m_type[source_node] = 3
        elif (relation == 6):
            m_type[source_node] = 0
        elif (relation == 7):
            m_type[source_node] = 4
        else:
            pass

paper_labels = {}
with open('../data/aminer/label.dat') as infile:  # id, author 唯一
    for line in infile.readlines():
        paper, label = line.strip().split(' ')[:2]
        paper = int(paper)
        label = int(label)
        paper_labels[paper] = label


rev = utils.reverse_type(m_type)
print(len(rev[0]))
print(len(rev[1]))
print(len(rev[2]))
print(len(rev[3]))

print("**************")



# for node in rev[1]:
#     labels = set()
#     for node_nn in graph[node][1]:
#         labels.add(paper_labels[node_nn])
#     if(len(labels) == 3):
#         print(node)

author = [8601, 10001, 10337, 11068, 11576, 11971, 20228, 31443]

visits = set()

for node in author:
    visits.add(node)
    for node_nn in graph[node][1]:
        visits.add(node_nn)


node_id = {}
for idx, node in enumerate(visits):
    node_id[node] = idx
    if(idx == 38):print(node)
print("all_cnt: ", len(node_id))

print(visits)

links = []
for node in visits:
    if m_type[node] == 0:
        for node_nn in graph[node][0]:
            if (node_nn not in visits):continue
            r = t_relation[m_type[node]][m_type[node_nn]]
            x = node_id[node]
            y = node_id[node_nn]
            links.append([x, y, r])
            links.append([y, x, r ^ 1])

# print(links)
# exit(0)

labels = []
for node in node_id.keys():
    if (m_type[node] == 0):
        if (node not in paper_labels):
            continue
        else:
            labels.append([node_id[node], paper_labels[node]])

ids = []

for node in node_id.keys():
    ids.append([node, node_id[node]])

np.savetxt("../decline/aminer/links.dat", np.array(links), fmt="%d")
np.savetxt("../decline/aminer/pre_label.dat", np.array(labels), fmt="%d")
np.savetxt("../decline/aminer/ids.dat", np.array(ids), fmt="%d")













