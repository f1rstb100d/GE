import networkx as nx
import random
import copy
from gensim.models import Word2Vec
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

class RandomWalker:
    def __init__(self, G):
        self.G = G

    def deepwalk_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        nodes = list(G.nodes())
        results = self._simulate_walks(nodes, num_walks, walk_length)
        return results

    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
        return walks


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.walker = RandomWalker(graph)
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter
        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model
        return model

    def get_embeddings(self):
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]
        return self._embeddings

def split_train_test_graph(input_edgelist, seed, testing_ratio=0.01, weighted=False):
    if (weighted):
        G = nx.read_weighted_edgelist(input_edgelist)
    else:
        G = nx.read_edgelist(input_edgelist, create_using = nx.DiGraph(), nodetype = None)
    node_num1, edge_num1 = len(G.nodes), len(G.edges)
    print('Original Graph: nodes:', node_num1, 'edges:', edge_num1)
    testing_edges_num = int(len(G.edges) * testing_ratio)
    random.seed(seed)
    testing_pos_edges = random.sample(G.edges, testing_edges_num)
    G_train = copy.deepcopy(G)
    for edge in testing_pos_edges:
        node_u, node_v = edge
        if (G_train.degree(node_u) > 1 and G_train.degree(node_v) > 1):
            G_train.remove_edge(node_u, node_v)
    G_train.remove_nodes_from(nx.isolates(G_train))
    node_num2, edge_num2 = len(G_train.nodes), len(G_train.edges)
    assert node_num1 == node_num2
    train_graph_filename = 'graph_train.edgelist'
    if weighted:
        nx.write_edgelist(G_train, train_graph_filename, data=['weight'])
    else:
        nx.write_edgelist(G_train, train_graph_filename, data=False)
    node_num1, edge_num1 = len(G_train.nodes), len(G_train.edges)
    print('Training Graph: nodes:', node_num1, 'edges:', edge_num1)
    return G, G_train, testing_pos_edges, train_graph_filename

def generate_neg_edges(original_graph, testing_edges_num, seed):
    L = list(original_graph.nodes())
    # create a complete graph
    G = nx.Graph()
    G.add_nodes_from(L)
    G.add_edges_from(itertools.combinations(L, 2))
    # remove original edges
    G.remove_edges_from(original_graph.edges())
    random.seed(seed)
    neg_edges = random.sample(G.edges, testing_edges_num)
    return neg_edges

def LinkPrediction(embedding_look_up, original_graph, train_graph, test_pos_edges, seed):
    random.seed(seed)
    train_neg_edges = generate_neg_edges(original_graph, len(train_graph.edges()), seed)
    # create a auxiliary graph to ensure that testing negative edges will not used in training
    G_aux = copy.deepcopy(original_graph)
    G_aux.add_edges_from(train_neg_edges)
    test_neg_edges = generate_neg_edges(G_aux, len(test_pos_edges), seed)
    # construct X_train, y_train, X_test, y_test
    X_train = []
    y_train = []
    for edge in train_graph.edges():
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(1)
    for edge in train_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(0)
    X_test = []
    y_test = []
    for edge in test_pos_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_test.append(feature_vector)
        y_test.append(1)
    for edge in test_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_test.append(feature_vector)
        y_test.append(0)
    # shuffle for training and testing
    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)
    c = list(zip(X_test, y_test))
    random.shuffle(c)
    X_test, y_test = zip(*c)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    clf1 = LogisticRegression(random_state=seed, solver='lbfgs')
    clf1.fit(X_train, y_train)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    y_pred = clf1.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('#' * 9 + ' Link Prediction Performance ' + '#' * 9)
    print(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
    print('#' * 50)
    return auc_roc, auc_pr, accuracy, f1

if __name__ == "__main__":
    G, G_train, testing_pos_edges, train_graph_filename = split_train_test_graph('Wiki_edgelist.txt', seed=0, weighted=False)
    model = DeepWalk(nx.read_edgelist('graph_train.edgelist', create_using=nx.DiGraph(), nodetype=None), walk_length=10, num_walks=80)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()
    result = LinkPrediction(embeddings, G, G_train, testing_pos_edges, seed=0)