import numpy as np
import random


class Alise():
    def __init__(self, G):
        self.g = G
        self.pj = {}
        self.pq = {}
        for i in range(np.size(self.g,axis=0)):
            self.pj[i], self.pq[i] = alias_setup(self.g[i,:])



def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            # print(kk)
            smaller.append(kk)
        else:
            larger.append(kk)
    # print("----------")
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        # print(small, large)
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

if __name__ == '__main__':
    g = np.random.rand(1,4)
    sum = np.sum(g)
    g = g / sum
    print(g)
    alias = Alise(g)
    print(alias.pj)
    print(alias.pq)
    print(alias_draw(alias.pj[0],alias.pq[0]))