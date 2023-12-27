import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# add either documents or terms to a topic's node in the PFNET, whatever the topic is associated with
def add_components_to_topic(PFNET, nmf_matrix, n_components, n_top_components, feature_names):
    top_components = []
    for topic_idx in range(n_components):
        top_components_for_topic = np.argsort(nmf_matrix[topic_idx])[::-1][:n_top_components] # apparently ::-1 is some weird magic that reverses the list
        top_components.append(top_components_for_topic)
    word_labels = {} 
    for topic_idx, components in enumerate(top_components):
        for component_idx in components:
            new_node = PFNET.number_of_nodes()
            PFNET.add_node(new_node)
            PFNET.add_edge(topic_idx, new_node)
            word_labels[new_node] = feature_names[component_idx]
            
    nx.set_node_attributes(PFNET, word_labels, 'Label')

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots((len(model.components_) - 1) // 5 + 1, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig('n100t10.png')
    plt.clf() 
    #plt.show()

def get_tfidf(data_samples, n_features):
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
    ) # if we change the df, we could get rid of even more words, worth experimenting
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    return tfidf, tfidf_vectorizer
    
def get_nmf(tfidf, n_components):
    nmf = NMF(
        n_components=n_components,
        random_state=1,
        
        init="nndsvda",
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ).fit(tfidf)
    return nmf
