import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Create a PFNET from the cosine similarity matrix using FASTPathfinder
def create_pfnet(G, q, r):
    s = G.number_of_nodes()
    weights_init = np.zeros((s,s))
    weights = np.zeros((s,s))
    hops = np.zeros((s,s))
    pfnet = np.zeros((s,s))

    for i, j, d in G.edges(data=True):
        weights_init[i,j] = d['weight']
        weights_init[j,i] = d['weight']

    for i in range(s):
        for j in range(s):
            weights[i,j] = weights_init[i,j] # this used to be -weights_init[i,j]
            if i==j:
                hops[i,j] = 0
            else:
                hops[i,j] = 1

    def update_weight_maximum(i, j, k, wik, wkj, weights, hops, p):
        if p<=q:
            if r==0:
                # r == infinity
                dist = min(wik, wkj) # this used to be max(wik, wkj)
            else:
                dist = (wik**r + wkj**r) ** (1/r)

            if dist > weights[i,j]: # this used to be dist < weights[i,j]
                weights[i,j] = dist
                weights[j,i] = dist
                hops[i,j] = p
                hops[j,i] = p
                
    def is_equal(a, b):
        return abs(a-b)<0.00001

    for k in range(s):
        for i in range(s):
            if i != k:
                beg = i + 1
                for j in range(beg, s):
                    if j != k:
                        update_weight_maximum(i, j, k, weights_init[i, k], weights_init[k, j], weights, hops, 2)
                        update_weight_maximum(i, j, k, weights[i, k], weights[k, j], weights, hops, hops[i, k] + hops[k, j])

    for i in range(s):
        for j in range(s):
            if not np.isclose(weights_init[i, j], 0):
                if np.isclose(weights[i, j], weights_init[i, j]):
                    pfnet[i, j] = weights_init[i, j]
                else:
                    pfnet[i, j] = 0

    return nx.from_numpy_array(pfnet)
