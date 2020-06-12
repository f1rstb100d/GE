# from stellargraph.datasets import IAEnronEmployees
from sklearn.model_selection import train_test_split
from stellargraph import StellarDiGraph
import pandas as pd
import numpy as np
import random
from stellargraph.data import TemporalRandomWalk
# from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
# from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import StandardScaler

# dataset = IAEnronEmployees()
# full_graph, edges = dataset.load() #50572
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

def positive_and_negative_links1(g, edges1, edges2):
    pos1 = list(edges1[["source", "target"]].itertuples(index=False))
    pos_all = pd.concat([edges1, edges2], ignore_index=True)
    # pos2 = list(edges2[["source", "target"]].itertuples(index=False))
    neg = sample_negative_examples1(g, pos_all)
    return pos1, neg


def sample_negative_examples1(g, positive_examples):
    positive_set = set(positive_examples)

    def valid_neg_edge(src, tgt):
        return (
            # no self-loops
            src != tgt
            and
            # neither direction of the edge should be a positive one
            (src, tgt) not in positive_set
            # and (tgt, src) not in positive_set
        )

    possible_neg_edges = [
        (src, tgt) for src in g.nodes() for tgt in g.nodes() if valid_neg_edge(src, tgt)
    ]

    return random.sample(possible_neg_edges, k=len(edges_train))

def positive_and_negative_links2(g, edges1, edges2):
    # pos1 = list(edges1[["source", "target"]].itertuples(index=False))
    pos_all = pd.concat([edges1, edges2], ignore_index=True)
    pos2 = list(edges2[["source", "target"]].itertuples(index=False))
    neg = sample_negative_examples2(g, pos_all)
    return pos2, neg


def sample_negative_examples2(g, positive_examples):
    positive_set = set(positive_examples)

    def valid_neg_edge(src, tgt):
        return (
            # no self-loops
            src != tgt
            and
            # neither direction of the edge should be a positive one
            (src, tgt) not in positive_set
            # and (tgt, src) not in positive_set
        )

    possible_neg_edges = [
        (src, tgt) for src in g.nodes() for tgt in g.nodes() if valid_neg_edge(src, tgt)
    ]

    return random.sample(possible_neg_edges, k=len(edges_test))


pos, neg = positive_and_negative_links1(graph, edges_train, edges_test)
pos_test, neg_test = positive_and_negative_links2(graph, edges_train, edges_test)

print(
    f"{graph.info()}\n"
    f"Training examples: {len(pos)} positive links, {len(neg)} negative links\n" # 9482
    f"Test examples: {len(pos_test)} positive links, {len(neg_test)} negative links" # 3161
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
t = 0
for i in temporal_walks:
    t += len(i)
print(t)

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



def operator_l2(u, v):
    # return (u - v) ** 2
    return np.concatenate((u, v), axis=0)

binary_operator = operator_l2


def link_examples_to_features(link_examples, transform_node):
    op_func = (
        operator_func[binary_operator]
        if isinstance(binary_operator, str)
        else binary_operator
    )
    return [op_func(transform_node(src), transform_node(dst)) for src, dst in link_examples]



def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

def labelled_links(positive_examples, negative_examples):
    return (positive_examples + negative_examples, np.repeat([1, 0], [len(positive_examples), len(negative_examples)]))

link_examples, link_labels = labelled_links(pos, neg)
link_examples_test, link_labels_test = labelled_links(pos_test, neg_test)
temporal_clf = LogisticRegression(solver='lbfgs', max_iter=5000)
temporal_link_features = link_examples_to_features(link_examples, temporal_embedding)
temporal_link_features_test = link_examples_to_features(link_examples_test, temporal_embedding)
temporal_clf.fit(temporal_link_features, link_labels)
temporal_score = evaluate_roc_auc(temporal_clf, temporal_link_features_test, link_labels_test)
print(f"Score (ROC AUC): {temporal_score:.2f}")

