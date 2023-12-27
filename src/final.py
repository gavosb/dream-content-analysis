

'''
 Topic Modeling on the Dream Bank dataset with NMF.
 Pruned by Pathfinder and saved for visualization.
 
 Author: Gavin Osborn
 Date: 12/13/23
'''

from time import time
import os.path
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from preprocess import *
from factorizations import *
from pfnets import *

n_samples = 20000 # n max words to consider overall
n_features = 100 # the n most significant words in the journal
n_components = 10 # n topics
n_top_words = 20 # for visualization

'''
PREPROCESSING
'''
data = create_data(0) # get data from first journal entry

data_samples = data[:n_samples]

'''
BAG-OF-WORDS WORD EMBEDDINGS
'''

tfidf, tfidf_vectorizer = get_tfidf(data_samples, n_features)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

'''
NMF MODELS
'''

nmf = get_nmf(tfidf, n_components)

plot_top_words(
    nmf, tfidf_feature_names, n_top_words, "Topics in NMF model"
)

# get component matrices #

W = nmf.fit_transform(tfidf); # document x topic
H = nmf.components_; # topic x term 
H_T = H.transpose() # term x topic
W_T = W.transpose() # topic x document

# get symmetric similarity matrices #

H_COS = cosine_similarity(H) # topics (by word), so clusters of words
W_T_COS = cosine_similarity(W_T) # topics (by document), so clusters of documents
H_T_COS = cosine_similarity(H_T) # words by topic relations
W_COS = cosine_similarity(W) # documents by topic relations

'''
GRAPHING
'''

# setup networkx graphs #

G_W_COS = nx.from_numpy_array(W_COS)
G_H_COS = nx.from_numpy_array(H_COS)
G_H_T_COS = nx.from_numpy_array(H_T_COS)
G_W_T_COS = nx.from_numpy_array(W_T_COS)

# save similarity matrices #

np.savetxt('terms.txt', tfidf_feature_names, fmt="%s", delimiter=' ',  newline="\n")
np.savetxt('W_matrix.txt', W_COS, fmt="%f", delimiter=' ',  newline="\n")
np.savetxt('W_T_matrix.txt', W_T_COS, fmt="%f", delimiter=' ',  newline="\n")
np.savetxt('H_matrix.txt', H_COS, fmt="%f", delimiter=' ',  newline="\n")
np.savetxt('H_T_matrix.txt', H_T_COS, fmt="%f", delimiter=' ',  newline="\n")
np.savetxt('docs.txt', list(range(W_COS.shape[0])), fmt="%d", delimiter=' ',  newline="\n")

# run pathfinder #

PFNET_G_W_COS = create_pfnet(G_W_COS, G_W_COS.number_of_nodes()-1, 0)
PFNET_G_W_T_COS = create_pfnet(G_W_T_COS, G_W_T_COS.number_of_nodes()-1, 0)
PFNET_G_H_COS = create_pfnet(G_H_COS, G_H_COS.number_of_nodes()-1, 0)
PFNET_G_H_T_COS = create_pfnet(G_H_T_COS, G_H_T_COS.number_of_nodes()-1, 0)

word_labels = {node: tfidf_feature_names[node] for node in PFNET_G_H_T_COS.nodes()}
nx.set_node_attributes(PFNET_G_H_T_COS, word_labels, 'Label')

add_components_to_topic(PFNET_G_H_COS, H_COS, n_components, n_top_words, tfidf_feature_names) # add words to topics
add_components_to_topic(PFNET_G_W_T_COS, H_COS, n_components, n_top_words, list(range(W_COS.shape[0]))) # add documents to topics

# save PFNETs as dot files #

current_dir = os.getcwd()
nx.nx_agraph.write_dot(PFNET_G_W_COS , f'{current_dir}/PFNET_G_W_COS.dot')
nx.nx_agraph.write_dot(PFNET_G_W_T_COS, f'{current_dir}/PFNET_G_W_T_COS.dot')
nx.nx_agraph.write_dot(PFNET_G_H_COS, f'{current_dir}/PFNET_G_H_COS.dot')
nx.nx_agraph.write_dot(PFNET_G_H_T_COS, f'{current_dir}/PFNET_G_H_T_COS.dot')

