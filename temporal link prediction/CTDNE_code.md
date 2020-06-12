Paper: [Continuous-Time Dynamic Network Embeddings](https://dl.acm.org/doi/pdf/10.1145/3184558.3191526)

```python
from sklearn.model_selection import train_test_split # 0.22.1
from stellargraph import StellarDiGraph # 1.1.0
import pandas as pd # 1.0.3
import numpy as np # 1.18.1
import random
from stellargraph.data import TemporalRandomWalk
from gensim.models import Word2Vec # 3.8.0

edges = pd.read_csv(
            'ia-enron-employees.edges',
            sep=" ",
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
        )
edges[["source", "target"]] = edges[["source", "target"]].astype(str)
nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )
full_graph, edges = StellarDiGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


# subset of edges to split
train_subset = 0.25
test_subset = 0.25

# number of edges to be kept in the graph
num_edges_graph = int(len(edges) * (1 - train_subset)) # 37929

# keep older edges in graph, and predict more recent edges
edges_graph = edges[:num_edges_graph] # 37929
edges_other = edges[num_edges_graph:] # 12643

# split recent edges further to train and test sets
edges_train, edges_test = train_test_split(edges_other, test_size=test_subset)

print(
    f"Number of edges in graph: {len(edges_graph)}\n"
    f"Number of edges in training set: {len(edges_train)}\n" # 9482
    f"Number of edges in test set: {len(edges_test)}" # 3161
)

graph = StellarDiGraph(
    nodes=pd.DataFrame(index=full_graph.nodes()),
    edges=edges_graph,
    edge_weight_column="time",
)



num_walks_per_node = 10
walk_length = 80
context_window_size = 10

num_cw = len(graph.nodes()) * num_walks_per_node * (walk_length - context_window_size + 1) # 107210

temporal_rw = TemporalRandomWalk(graph)
temporal_walks = temporal_rw.run(
    num_cw=num_cw,
    cw_size=context_window_size,
    max_walk_length=walk_length,
    walk_bias="exponential",
)

print("Number of temporal random walks: {}".format(len(temporal_walks))) # 1792

embedding_size = 128
temporal_model = Word2Vec(
    temporal_walks,
    size=embedding_size,
    window=context_window_size,
    min_count=0,
    sg=1,
    workers=2,
    iter=1,
)

unseen_node_embedding = np.zeros(embedding_size)


def temporal_embedding(u):
    try:
        return temporal_model.wv[u]
    except KeyError:
        return unseen_node_embedding


print(temporal_embedding('151'))
```

Parameter:
`num_walks_per_node = 10`
`walk_length = 80`
`context_window_size = 10`
`initial_edge_bias = None (uniform distribution)`
`walk_bias="exponential"`
`embedding_size = 128`

Main idea:
1. 数据集按时间增加排列，扣出前75%作为动态图构建，后25%中随机选择75%作为训练集。最后剩下的25%为测试集。

2. 随机选择初始的边，以下一条边的时间戳必须大于等于（stellargraph源码中只考虑了大于），选下一个节点概率符合exponential分布，直到达到最大长度或没有下一条边结束。

3. 将生成的句子用Word2vec训练得到每个单词也就是节点的向量。（stellargraph只规定了总的用来训练的窗口数量，而没有规定以每个节点开始进行多少次的随机游走）

[code reference](https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/ctdne-link-prediction.html) and [dataset](http://nrvis.com/download/data/dynamic/ia-enron-employees.zip)