Paper: [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)


```python
import networkx as nx # ver = 2.3
from gensim.models import Word2Vec # ver = 3.8.1
import random


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


G = nx.read_edgelist('Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None)
model = DeepWalk(G, walk_length=10, num_walks=80)
model.train(window_size=5, iter=3)
embeddings = model.get_embeddings()
print(embeddings['2404'])
```

Parameter:
`walk_length = 10`
`num_walks = 80 (walks per vertex)`
`embedding_size = 128`
`window_size = 5`

Main idea:
1. Start with each node (num_walks) times with the maximum length randomwalk lenth (walk_length), Then we get N * (num_walks) sentences with max length (walk_length).

2. Use Word2Vec (skip-gram) to train the model with all sentences (corpus).

3. Get (embedding_size) dimensional vector for each node.

[code reference](https://github.com/shenweichen/GraphEmbedding) and [dataset](https://github.com/thunlp/OpenNE/tree/master/data/wiki/Wiki_edgelist.txt)