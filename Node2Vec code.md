Paper: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf)
```python
import networkx as nx # ver = 2.3
import numpy as np
from gensim.models import Word2Vec # ver = 3.8.1
import random

def alias_sample(accept, alias):
    """
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

    def get_alias_edge(self, t, v): # link from t to v
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

if __name__ == "__main__":
    G=nx.read_edgelist('Wiki_edgelist.txt', create_using = nx.DiGraph(), nodetype = None)
    model=Node2Vec(G, walk_length = 10, num_walks = 80, p = 0.25, q = 4, workers = 1)
    model.train(window_size = 5, iter = 3)
    embeddings=model.get_embeddings()
    print(embeddings['2404'])
```

Parameter:
`walk_length = 10`
`num_walks = 80 (walks per vertex)`
`p = 0.25`
`q = 4`
`embedding_size = 128`
`window_size = 5`

Main idea:
1. 对每一个节点创建它与其邻居的访问概率，由于边没有权值，所以访问下一个邻居的概率为$\frac{1}{nbr}$， 然后根据此概率创建节点的alias表，返回一个accept和alias![](https://github.com/f1rstb100d/GE/blob/master/png/alias_table.png)，两个数组，一个里面存着第i列对应的事件i矩形站的面积百分比【也即其概率】，上图的话数组就为Prab[$\frac{2}{3}$, $1$, $\frac{1}{3}$, $\frac{1}{3}$]，另一个数组里面储存着第i列不是事件i的另外一个事件的标号，像上图就是Alias[2, NULL, 1, 1] 。
2. 对有向图每一对边，同样计算边的结束节点的邻居的概率![](https://github.com/f1rstb100d/GE/blob/master/png/node2vec_pq.png)，已有t到v边，找到所有v的邻居，如果还是t那么概率为$\frac{1}{p}$，如果节点(x1)直接和t相连那么概率为$1$，如果到t的路径距离大于1(x2, x3)那么概率为$\frac{1}{q}$，同样将这个下一个节点选择概率list转换成alias_table的accept, alias格式。
3. 然后以每个节点为起点进行num_walks次的随机游走，截取walk_length长度的游走路径(可以小于walk_length)。其中刚选定了第一个点，那么下一个点的选择使用步骤1中的概率表，选定完第二个点之后就会有一条路径，那么再下一个点以及之后的所有点都使用步骤2中的概率表。
4. 最后使用Word2vec中的skip-gram模型训练生成的所有随机游走路径(句子)，得到每个节点的表示向量。

[code reference](https://github.com/shenweichen/GraphEmbedding) and [dataset](https://github.com/thunlp/OpenNE/tree/master/data/wiki/Wiki_edgelist.txt)
[alias采样](https://blog.csdn.net/haolexiao/article/details/65157026)