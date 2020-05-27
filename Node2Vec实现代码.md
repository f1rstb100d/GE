Paper: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf)
```python
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import random
import os


def alias_sample(accept, alias):
    """
    https://blog.csdn.net/haolexiao/article/details/65157026
    :param accept: 每一列下面那项的概率
    :param alias: 每一列上面那项的编号
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N) #第一次随机第几列
    r = np.random.random() # 第二次随机选上面的还是下面的
    if r < accept[i]:
        return i
    else:
        return alias[i]


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
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



class RandomWalker:
    def __init__(self, G, p=1, q=1):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break # 不一定每次随机游走都达到最长路径

        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G
        nodes = list(G.nodes())
        walks = self._simulate_walks(nodes, num_walks, walk_length)
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs) # {'1397': ([1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0])}

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1]) # {('1397', '1470'): ([1, 0.9032258064516129, 0.22580645161290322, 0.9032258064516129, 0.9032258064516129, 0.22580645161290322, 0.22580645161290322], [0, 0, 0, 0, 0, 0, 0])}

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return



class Node2Vec:
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
        kwargs["sg"] = 1
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


if __name__ == "__main__":
    G=nx.read_edgelist('Wiki_edgelist.txt', create_using = nx.DiGraph(), nodetype = None)
    model=Node2Vec(G, walk_length = 10, num_walks = 80, p = 0.25, q = 4, workers = 1)
    model.train(window_size = 5, iter = 3)
    embeddings=model.get_embeddings()
    print(embeddings['2404'])

```

[reference](https://github.com/shenweichen/GraphEmbedding) and [dataset](https://github.com/thunlp/OpenNE/tree/master/data/wiki/Wiki_edgelist.txt)