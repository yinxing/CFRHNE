"""
按照 8:2 的比例划分训练集和测试集
"""
import random
import os

# dblp_file = '../data/dblp/links.dat'

dataset = "aminer"
yelp_file = '../data/' + dataset + '/links.dat'


def split_lines(filename, output_dir, output_name):
    train = []
    test = []
    with open(filename) as infile:
        for line in infile.readlines():
            if random.random() < 0.8:
                train.append(line.strip())
            else:
                test.append(line.strip())

    d = os.path.join('../data', output_dir)
    if not os.path.exists(d):
        os.mkdir(d)

    train_file = os.path.join(d, '%s.train_0.8' % output_name)
    test_file = os.path.join(d, '%s.test_0.2' % output_name)

    train_str = '\n'.join(train)
    f = open(train_file, 'w', encoding='utf-8')
    f.write(train_str)
    f.close()

    test_str = '\n'.join(test)
    f = open(test_file, 'w', encoding='utf-8')
    f.write(test_str)
    f.close()


# split_lines(dblp_file, 'dblp', 'dblp_ap')
split_lines(yelp_file, dataset, dataset + '_ap')
