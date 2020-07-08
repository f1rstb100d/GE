
from sklearn.model_selection import train_test_split
from stellargraph import StellarDiGraph
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from stellargraph.random import random_state
from stellargraph.core.graph import StellarGraph
from stellargraph.core.schema import GraphSchema
from scipy import stats
from scipy.special import softmax


edges = pd.read_csv(
            '../combinelink.txt',
            sep=" ",
            header=None,
            names=["source", "target", "time"],
            usecols=["source", "target", "time"],
        )
edges[["source", "target"]] = edges[["source", "target"]].astype(str)
nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

full_graph, edges = StellarDiGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


graph = StellarDiGraph(
    nodes=pd.DataFrame(index=full_graph.nodes()),
    edges=edges,
    edge_weight_column="time",
)



num_walks_per_node = 10
walk_length = 80
context_window_size = 10

num_cw = len(graph.nodes()) * num_walks_per_node * (walk_length - context_window_size + 1) # 633*10*71 = 449430

print(num_cw)


def naive_weighted_choices(rs, weights):
    probs = np.cumsum(weights)
    idx = np.searchsorted(probs, rs.random() * probs[-1], side="left")

    return idx



class TemporalRandomWalk():

    def __init__(self,graph):
        self.graph=graph
        self.cw_size = 10
        self.max_walk_length = 80
        self.num_cw = num_cw
        self.initial_edge_bias = None
        self.walk_bias = "exponential"
        self.p_walk_success_threshold = 0.01
        self.seed = None
        self.graph_schema = None
        self._random_state, self._np_random_state = random_state(self.seed)
        if not isinstance(self.graph, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph.")

        if not self.graph_schema:
            self.graph_schema = self.graph.create_graph_schema()
        else:
            self.graph_schema = self.graph_schema

        if type(self.graph_schema) is not GraphSchema:
            self._raise_error(
                "The parameter graph_schema should be either None or of type GraphSchema."
            )



    def run(self):

        np_rs = self._np_random_state if self.seed is None else np.random.RandomState(self.seed)
        walks = []
        num_cw_curr = 0

        sources, targets, _, times = self.graph.edge_arrays(include_edge_weight=True)
        edge_biases = self._temporal_biases(
            times, None, bias_type=self.initial_edge_bias, is_forward=False,
        )

        successes = 0
        failures = 0

        def not_progressing_enough():
            # Estimate the probability p of a walk being long enough; the 95% percentile is used to
            # be more stable with respect to randomness. This uses Beta(1, 1) as the prior, since
            # it's uniform on p
            posterior = stats.beta.ppf(0.95, 1 + successes, 1 + failures)
            return posterior < self.p_walk_success_threshold

        # loop runs until we have enough context windows in total
        while num_cw_curr < self.num_cw:
            print(num_cw_curr)
            first_edge_index = self._sample(len(times), edge_biases, np_rs)
            src = sources[first_edge_index]
            dst = targets[first_edge_index]
            t = times[first_edge_index]

            remaining_length = self.num_cw - num_cw_curr + self.cw_size - 1

            walk = self._walk(
                src, dst, t, min(self.max_walk_length, remaining_length), self.walk_bias, np_rs
            )
            if len(walk) >= self.cw_size:
                walks.append(walk)
                num_cw_curr += len(walk) - self.cw_size + 1
                successes += 1
            else:
                failures += 1
                if not_progressing_enough():
                    raise RuntimeError(
                        f"Discarded {failures} walks out of {failures + successes}. "
                        "Too many temporal walks are being discarded for being too short. "
                        f"Consider using a smaller context window size (currently cw_size={self.cw_size})."
                    )

        return walks

    def _sample(self, n, biases, np_rs):
        if biases is not None:
            assert len(biases) == n
            return naive_weighted_choices(np_rs, biases)
        else:
            return np_rs.choice(n)

    def _exp_biases(self, times, t_0, decay):
        # t_0 assumed to be smaller than all time values
        return softmax(t_0 - np.array(times) if decay else np.array(times) - t_0)

    def _temporal_biases(self, times, time, bias_type, is_forward):
        if bias_type is None:
            # default to uniform random sampling
            return None

        # time is None indicates we should obtain the minimum available time for t_0
        t_0 = time if time is not None else min(times)

        if bias_type == "exponential":
            # exponential decay bias needs to be reversed if looking backwards in time
            return self._exp_biases(times, t_0, decay=is_forward)
        else:
            raise ValueError("Unsupported bias type")

    def _step(self, node, time, bias_type, np_rs):
        """
        Perform 1 temporal step from a node. Returns None if a dead-end is reached.
        """
        neighbours, times = self.graph.neighbor_arrays(node, include_edge_weight=True)
        neighbours = neighbours[times >= time]
        times = times[times >= time]

        if len(neighbours) > 0:
            biases = self._temporal_biases(times, time, bias_type, is_forward=True)
            chosen_neighbour_index = self._sample(len(neighbours), biases, np_rs)
            next_node = neighbours[chosen_neighbour_index]
            next_time = times[chosen_neighbour_index]
            return next_node, next_time
        else:
            return None

    def _walk(self, src, dst, t, length, bias_type, np_rs):
        walk = [src, dst]
        node, time = dst, t
        for _ in range(length - 2):
            result = self._step(node, time=time, bias_type=bias_type, np_rs=np_rs)

            if result is not None:
                node, time = result
                walk.append(node)
            else:
                break

        return walk


temporal_rw = TemporalRandomWalk(graph)
temporal_walks = temporal_rw.run()

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



edges_0620 = pd.read_csv(
            '../relink0620.txt',
            sep=" ",
            header=None,
            names=["source", "target",],
            usecols=["source", "target"],
        )
edges_0620[["source", "target"]] = edges_0620[["source", "target"]].astype(str)
full_graph_0620, edges_0620 = StellarDiGraph(nodes=nodes, edges=edges_0620), edges_0620


edges_train_pos, edges_test_pos = train_test_split(edges_0620, test_size=0.2)




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

    return random.sample(possible_neg_edges, k=len(edges_train_pos))



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

    return random.sample(possible_neg_edges, k=len(edges_test_pos))



pos_train, neg_train = positive_and_negative_links1(full_graph_0620, edges_train_pos, edges_test_pos)
pos_test, neg_test = positive_and_negative_links2(full_graph_0620, edges_train_pos, edges_test_pos)

print(
    f"{graph.info()}\n"
    f"Training examples: {len(pos_train)} positive links, {len(neg_train)} negative links\n" # 9482
    f"Test examples: {len(pos_test)} positive links, {len(neg_test)} negative links" # 3161
)




def labelled_links(positive_examples, negative_examples):
    return (positive_examples + negative_examples, np.repeat([1, 0], [len(positive_examples), len(negative_examples)]))



link_examples_train, link_labels_train = labelled_links(pos_train, neg_train)
link_examples_test, link_labels_test = labelled_links(pos_test, neg_test)


unseen_node_embedding = np.zeros(embedding_size)


def temporal_embedding(u):
    try:
        return temporal_model.wv[u]
    except KeyError:
        return unseen_node_embedding


def operator_l2(u, v):
    return np.concatenate((u, v), axis=0)

binary_operator = operator_l2


def link_examples_to_features(link_examples, transform_node):
    op_func = (
        operator_func[binary_operator]
        if isinstance(binary_operator, str)
        else binary_operator
    )
    return [op_func(transform_node(src), transform_node(dst)) for src, dst in link_examples]



temporal_clf = LogisticRegression(solver='lbfgs', max_iter=5000)
temporal_link_features_train = link_examples_to_features(link_examples_train, temporal_embedding)
temporal_link_features_test = link_examples_to_features(link_examples_test, temporal_embedding)

temporal_clf.fit(temporal_link_features_train, link_labels_train)


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


temporal_score = evaluate_roc_auc(temporal_clf, temporal_link_features_test, link_labels_test)
print(f"Score (ROC AUC): {temporal_score:.2f}")