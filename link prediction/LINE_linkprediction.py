import networkx as nx
import random
import copy
import math
import numpy as np
import itertools
import tensorflow as tf # version = 1.14.0
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx

def create_alias_table(area_ratio):
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l #list每一项乘2405
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)
    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    return accept, alias

def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]
    
def line_loss(y_true, y_pred):
    return -K.mean(K.log(K.sigmoid(y_true*y_pred)))

def create_model(numNodes, embedding_size, order='second'):
    v_i = Input(shape=(1,)) #输入是一维数组，数组中有1个元素
    v_j = Input(shape=(1,))
    first_emb = Embedding(numNodes, embedding_size, name='first_emb')
    second_emb = Embedding(numNodes, embedding_size, name='second_emb')# 该顶点本身的表示向量
    context_emb = Embedding(numNodes, embedding_size, name='context_emb')# 该点作为其他顶点的上下文(邻居)顶点时的表示向量
    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)
    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)
    first = Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-1, keep_dims=False), name='first_order')([v_i_emb, v_j_emb])
    second = Lambda(lambda x: tf.reduce_sum(x[0]*x[1], axis=-1, keep_dims=False), name='second_order')([v_i_emb_second, v_j_context_emb]) # 内积
    if order == 'first':
        output_list = [first]
    elif order == 'second':
        output_list = [second]
    else:
        output_list = [first, second]
    model = Model(inputs=[v_i, v_j], outputs=output_list)
    return model, {'first': first_emb, 'second': second_emb}
	
	
class LINE:
    def __init__(self, graph, embedding_size=8, negative_ratio=5, order='second',):
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be fisrt,second,or all')
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.use_alias = True
        self.rep_size = embedding_size
        self.order = order
        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order
        self.node_size = graph.number_of_nodes() # 2405
        self.edge_size = graph.number_of_edges() # 16523
        self.samples_per_epoch = self.edge_size*(1+negative_ratio) # 99138
        self._gen_sampling_table() # 顶点负采样和边采样需要的采样表
        self.reset_model()

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = ((self.samples_per_epoch - 1) // self.batch_size + 1)*times # 97

    def reset_model(self, opt='adam'):
        self.model, self.embedding_dict = create_model(self.node_size, self.rep_size, self.order)
        self.model.compile(opt, line_loss) # 默认learning rate
        self.batch_it = self.batch_iter(self.node2idx)

    def _gen_sampling_table(self):
        # create sampling table for vertex
        power = 0.75
        numNodes = self.node_size
        node_degree = np.zeros(numNodes)  # out degree
        node2idx = self.node2idx
        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)
        total_sum = sum([math.pow(node_degree[i], power) for i in range(numNodes)])
        norm_prob = [float(math.pow(node_degree[j], power)) / total_sum for j in range(numNodes)]
        self.node_accept, self.node_alias = create_alias_table(norm_prob)
        # create sampling table for edge
        numEdges = self.graph.number_of_edges() # 16523
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0) for edge in self.graph.edges()]) # 16523.0
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) * numEdges / total_sum for edge in self.graph.edges()] # 全部是1[1.0, 1.0, ..., 1.0]
        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

    def batch_iter(self, node2idx):
        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]
        data_size = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio # 6
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:
                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h))*-1
                t = []
                for i in range(len(h)):
                    t.append(alias_sample(self.node_accept, self.node_alias))
            if self.order == 'all':
                yield ([np.array(h), np.array(t)], [sign, sign])
            else:
                yield ([np.array(h), np.array(t)], [sign])
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)
            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)

    def get_embeddings(self,):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding
        return self._embeddings

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1, times=1):
        self.reset_training_config(batch_size, times)
        hist = self.model.fit_generator(self.batch_it, epochs=epochs, initial_epoch=initial_epoch, steps_per_epoch=self.steps_per_epoch, verbose=verbose) # params = <class 'dict'>: {'batch_size': None, 'epochs': 50, 'steps': 97, 'samples': 97, 'verbose': 2, 'do_validation': False, 'metrics': ['loss']}
        return hist

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=p, q=q)
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1 # skip-gram
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter
        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model
        return model

    def get_embeddings(self,):
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]
        return self._embeddings

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
    model = LINE(nx.read_edgelist('graph_train.edgelist', create_using=nx.DiGraph(), nodetype=None), embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()
    result = LinkPrediction(embeddings, G, G_train, testing_pos_edges, seed=0)