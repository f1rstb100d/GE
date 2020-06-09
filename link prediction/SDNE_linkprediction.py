import networkx as nx
import random
import numpy as np
import time
import copy
import itertools
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1_l2
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

def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
        b_ = np.ones_like(y_true)
        b_[y_true != 0] = beta
        x = K.square((y_true - y_pred) * b_)
        t = K.sum(x, axis=-1, )
        return K.mean(t)
    return loss_2nd

def l_1st(alpha):
    def loss_1st(y_true, y_pred):
        L = y_true
        Y = y_pred
        batch_size = tf.to_float(K.shape(L)[0])
        return alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y)) / batch_size
    return loss_1st

def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
    A = Input(shape=(node_size,))
    L = Input(shape=(None,))
    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2), name='1st')(fc)
        else:
            fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2))(fc)
    Y = fc
    for i in reversed(range(len(hidden_size) - 1)):
        fc = Dense(hidden_size[i], activation='relu', kernel_regularizer=l1_l2(l1, l2))(fc)
    A_ = Dense(node_size, 'relu', name='2nd')(fc)
    model = Model(inputs=[A, L], outputs=[A_, Y])
    emb = Model(inputs=A, outputs=Y)
    return model, emb

class SDNE(object):
    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4, ):
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)
        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.A, self.L = self._create_A_L(self.graph, self.node2idx)  # Adj Matrix,L Matrix
        self.reset_model()
        self.inputs = [self.A, self.L]
        self._embeddings = {}

    def reset_model(self, opt='adam'):
        self.model, self.emb_model = create_model(self.node_size, hidden_size=self.hidden_size, l1=self.nu1, l2=self.nu2)
        self.model.compile(opt, [l_2nd(self.beta), l_1st(self.alpha)])
        # self.get_embeddings()

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1):
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                    batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit([self.A.todense(), self.L.todense()], [self.A.todense(), self.L.todense()], batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, verbose=verbose, shuffle=False, )
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(i * batch_size, min((i + 1) * batch_size, self.node_size))
                    A_train = self.A[index, :].todense()
                    L_mat_train = self.L[index][:, index].todense()
                    inp = [A_train, L_mat_train]
                    batch_losses = self.model.train_on_batch(inp, inp)
                    losses += batch_losses
                losses = losses / steps_per_epoch
                logs['loss'] = losses[0]
                logs['2nd_loss'] = losses[1]
                logs['1st_loss'] = losses[2]
                epoch_time = int(time.time() - start_time)
                hist.on_epoch_end(epoch, logs)
                if verbose > 0:
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                    print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f}'.format(epoch_time, losses[0], losses[1], losses[2]))
            return hist

    def get_embeddings(self):
        self._embeddings = {}
        embeddings = self.emb_model.predict(self.A.todense(), batch_size=self.node_size)
        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[look_back[i]] = embedding
        return self._embeddings

    def _create_A_L(self, graph, node2idx):
        node_size = graph.number_of_nodes()
        A_data = []
        A_row_index = []
        A_col_index = []
        for edge in graph.edges():
            v1, v2 = edge
            edge_weight = graph[v1][v2].get('weight', 1)
            A_data.append(edge_weight)
            A_row_index.append(node2idx[v1])
            A_col_index.append(node2idx[v2])
		# A为邻接矩阵
        A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size)) # Compressed Sparse Row matrix 第i行第j列有边（i->j）矩阵就是1，没有就是0
		# A_为重构后的邻接矩阵
        A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)), shape=(node_size, node_size)) # list相加直接后面拼接，把A对角线折回来加上，矩阵里有0 1 2
        D = sp.diags(A_.sum(axis=1).flatten().tolist()[0]) # 按行加，放到对角线，度矩阵（由该点发出的边的数量）
        L = D - A_ # 拉普拉斯矩阵 https://blog.csdn.net/qq_30159015/article/details/83271065
        return A, L

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
    model = SDNE(nx.read_edgelist('graph_train.edgelist', create_using=nx.DiGraph(), nodetype=None), hidden_size=[256, 128],)
    model.train(batch_size=3000, epochs=40, verbose=2)
    embeddings = model.get_embeddings()
	# link prediction reference: https://github.com/xiangyue9607/BioNEV
    result = LinkPrediction(embeddings, G, G_train, testing_pos_edges, seed=0)