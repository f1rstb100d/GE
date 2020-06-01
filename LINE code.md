Paper: [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)
```python
import networkx as nx
import numpy as np
import math
import random
import tensorflow as tf # version = 1.14.0
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda
from tensorflow.python.keras.models import Model

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

if __name__ == "__main__":
    G = nx.read_edgelist('Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None)
    model = LINE(G, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()
    print(embeddings['2404'])
```

Parameter:
`embedding_size = 128`
`negative_ratio = 5`
`batch_size = 1024`
`epochs = 50`
`verbose = 1`

Main idea:
1. 计算每个节点的出度，然后将每个节点出度的0.75次方除以所有节点出度的0.75次方的和来均一化得到针对每个节点的概率list，然后构造成点的alias table(参考Node2vec)

2. 每条边的概率都是1，创建一个长度为边数内容全为1的list，然后以此概率构造边的alias table

3. 然后构造模型，输入为两个点的编号，然后embedding层得到向量，之后计算向量的内积，compile上自定义的损失函数（内积与真实标签1 -1的损失函数），所以模型是输入两个点编号输出内积，最后返回整个模型个第二层embedding层的表示向量

4. 构造训练用的batch，对所有边随机以1024（batch_size）为块进行划分，对每一块中1024条边通过edge的alias表进行一次正采样（就是这1024条边）并打上标签1，再进行5（negative_ratio）次负采样，对于之前存在的h到t的边，h固定，使用node的alias表随机一个新的t（大概率是不存在边的），并打上标签-1。返回1024![](http://latex.codecogs.com/gif.latex?*)6条边，这其中1024条label为1，1024![](http://latex.codecogs.com/gif.latex?*)5条label为0

5. 计算一下一个epoch中有97个batch，然后用采样结果进行模型训练，得到每个节点的128（embedding_size）维向量

[code reference](https://github.com/shenweichen/GraphEmbedding) and [dataset](https://github.com/thunlp/OpenNE/tree/master/data/wiki/Wiki_edgelist.txt)