Paper: [DynGEM: Deep Embedding Method for Dynamic Graphs](https://arxiv.org/pdf/1805.11273.pdf)

文档结构：
```
dyngem
├─ dyngem.py
├─ graph_pickler.py
├─ link_prediction.py
├─ models
│    ├─ prev_model_1.h5
│    ├─ prev_model_2.h5
│    ├─ prev_model_3.h5
│    ├─ prev_model_4.h5
│    ├─ prev_model_5.h5
│    ├─ prev_model_6.h5
│    └─ prev_model_7.h5
└─ snapshots
       ├─ s1.txt
       ├─ s1_graph.gpickle
       ├─ s2.txt
       ├─ s2_graph.gpickle
       ├─ s3.txt
       ├─ s3_graph.gpickle
       ├─ s4.txt
       ├─ s4_graph.gpickle
       ├─ s5.txt
       ├─ s5_graph.gpickle
       ├─ s6.txt
       ├─ s6_graph.gpickle
       ├─ s7.txt
       └─ s7_graph.gpickle

```

graph_pickler.py
```python
import numpy as np
import networkx as nx
for i in range(1,8):
	filename = 'snapshots/s' + str(i) + ".txt"
	nfilename = 'snapshots/s' + str(i) + "_graph.gpickle"
	G = nx.read_edgelist(filename, create_using=nx.Graph())
	print(G.number_of_nodes())
	nx.write_gpickle(G,nfilename)
```
snapshots下有8个文件，分别是是8个时间片段的截图，每行的格式为`node1 node2`，使用nx重新定义输入数据格式。


dyngem.py
```python
import networkx as nx # version=1.11
import keras # version<=2.2.4
from keras.models import Model, Sequential, load_model
from keras.layers  import Dense, Input, Embedding, Reshape,  Lambda
from keras import backend as K, regularizers
import numpy as np
from functools import reduce
# tensorflow version=1.13.1

dynamic_series = []
activation_fn = 'relu'
activation_fn_embedding_layer='relu'
loss_function = 'binary_crossentropy'
dynamic_model_build_epochs_number = 300

'''
#Loss Function for preserving First and Second Order
def build_reconstruction_loss(beta):
    """
    return the loss function for 2nd order proximity
    beta: the definition below Equation 3"""
    assert beta > 1

    def reconstruction_loss(true_y, pred_y):
        diff = K.square(true_y - pred_y)

        # borrowed from https://github.com/suanrong/SDNE/blob/master/model/sdne.py#L93
        weight = true_y * (beta - 1) + 1

        weighted_diff = diff * weight
        return K.mean(K.sum(weighted_diff, axis=1))  # mean square error
    return reconstruction_loss


def edge_wise_loss(true_y, embedding_diff):
    """1st order proximity
    """
    # true_y supposed to be None
    # we don't use it
    return K.mean(K.sum(K.square(embedding_diff), axis=1))  # mean square error
'''

#Loading Snapshot Pickled With networkx 1.11
def loadRealGraphSeries(file_prefix, startId, endId):
    graphs = []
    for file_id in range(startId, endId + 1):
        graph_file = file_prefix + str(file_id) + '_graph.gpickle'
        graphs.append(nx.read_gpickle(graph_file))
    return graphs

def get_encoder(model, input_name):
    #Encoder Model
    model = model.model
    encoder = Model(model.input, model.get_layer('embedding-layer').output)
    return encoder

def get_decoder(model):
    decoder = None
    return decoder

def link_prediction():
    return None

def get_embedding(encoder, graph_name, output_name):
    g = nx.read_edgelist(graph_name, create_using=nx.Graph())
    g = nx.convert_node_labels_to_integers(g)
    graph = g
    # N = graph.number_of_nodes()
    adj_mat = nx.adjacency_matrix(graph).toarray()
    embedding = encoder.predict(adj_mat)
    np.savetxt(output_name,embedding)
    return embedding

def build_model():
    #Main Function
    embedding_dim = 0 # 1/4 of number of nodes
    # Encoding Layer Dims
    encoding_dim =  []
    # Decoding Layer Dims
    decoding_dim = [] 
    encoding_layers = []
    decoding_layers = []
    dynamic_series = loadRealGraphSeries('snapshots/s',1,7)
    count = 0
    node_count = []
    #Initial Values
    beta=2
    alpha=2
    l2_param=1e-3
    print("Builiding Model.................")
    final_model = None
    for g in dynamic_series:
        N = g.number_of_nodes()
        node_count.append(N)
        count = count + 1
        adj_mat = nx.adjacency_matrix(g).toarray()
        edges = np.array(list(g.edges_iter()))
        weights = [ g[u][v].get('weight',1.0) for u,v in g.edges_iter() ]

        if count == 1 :
            # Create Model from Scratch
            embedding_dim = (int) (N/4) # 1/4 
            # Embedding Layers
            i =  embedding_dim
        
            while ((i+embedding_dim) < N):
                i = i + embedding_dim
                encoding_dim.append(i)         
            encoding_dim.append(N)
            decoding_dim = encoding_dim
            encoding_dim = encoding_dim[::-1]
            
            print("Number Of Nodes: ", end=" ")
            print(N)
            print("Decoding Dimensions : ", end=" ")
            print(decoding_dim)
            print("Encoding Dimensions : ", end=" ")
            print(encoding_dim)
            print("Embedding Dimension : ", end=" ")
            print(embedding_dim)

            #Initializing Model
            model = Sequential()

            i = 0
            for dim in encoding_dim:
                i = i + 1
                layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='encoding-layer-{}'.format(i))
                model.add(layer)

            
            model.add(Dense(embedding_dim, activation=activation_fn_embedding_layer, kernel_regularizer=regularizers.l2(l2_param), name='embedding-layer'))
            
            i = 0
            #decoding_dim.append(N)
            for dim in decoding_dim:
                i = i + 1
                layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='decoding-layer-{}'.format(i))
                model.add(layer)



            model.compile(loss=loss_function, optimizer='adam', metrics=['acc','mae'])
            model.fit(adj_mat,adj_mat,epochs=300)
            model_name = 'models/prev_model_{}.h5'.format(count)

            model.summary()
            model.save(model_name)

            print("SDNE Initial Model Built Completed ... ")
            final_model = model # Saving The Final Model

        else:
            # Create Model from Scratch
            print("Adding More Dynamic Layers {}".format(count))
            prev_N = node_count[count - 2]
            N = g.number_of_nodes()
            prev_model_name = 'models/prev_model_{}.h5'.format(count-1)
            curr_model_name = 'models/prev_model_{}.h5'.format(count)

            if prev_N == N:  
                prev_model = load_model(prev_model_name)
                # No need to add layers if number of nodes are same but just fit the new dataset
                prev_model.compile(loss=loss_function, optimizer='adam', metrics=['mae','acc'])
                prev_model.fit(adj_mat,adj_mat,epochs=dynamic_model_build_epochs_number)
                prev_model.save(curr_model_name)
                continue

            if prev_N < N:
                prev_model = load_model(prev_model_name)
            
                input_layer = Dense(N,input_dim=N,activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='dynamic-encoding-layer-{}'.format(count))
                input_layer_dummy = Dense(prev_N, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='dynamic-encoding-layer-support-{}'.format(count))
                output_layer = Dense(N, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='dynamic-decoding-layer-{}'.format(count))
                
                curr_model = Sequential()

                #Adding the New Input Layer
                curr_model.add(input_layer)
                curr_model.add(input_layer_dummy)

                #Adding the Existing Layers
                model_api = prev_model.model
                skip_input_layer=0 # This is to Skip the Previous input Layer
                for layer in model_api.layers:
                    if skip_input_layer == 0:
                        skip_input_layer = 1
                        continue
                    curr_model.add(layer)
            
                #Adding the Existing Layer
                curr_model.add(output_layer)

                curr_model.compile(loss=loss_function, optimizer='adam', metrics=['mae','acc'])
                curr_model.fit(adj_mat,adj_mat,epochs=100)

                #Sequential Before Saving
                curr_model.summary()
                curr_model.save(curr_model_name)
                print("Dynamic Layer {} Addition Completed .".format(count))

                #Assigning Final Model
                final_model = curr_model

    return final_model,'dynamic-encoding-layer-{}'.format(count) #Returning the final model
            
        
f_model, input_layer_name = build_model()
f_model.summary()

print("******************* Embedding ************************")

# encoder = get_encoder(f_model, input_layer_name)
# get_embedding(encoder,"graphs/final_graph.txt", "embedding/final_output.embedding")

model = load_model("models/prev_model_7.h5")
encoder = get_encoder(model, input_layer_name)
get_embedding(encoder,"snapshots/s7.txt", "final_output.embedding")
```
1. 图的节点数是递增的，第一个图有91个节点，所以构造一个层数为[91,88,66,44,22,44,66,88,91]的encoder和decoder全连接神经网络，其中有22个节点层为embedding的向量，输入和输出为同一个91x91的邻接矩阵，一次训练一个的话就会有91个batches，第一次将整个邻接矩阵训练300个epochs.

2. 第二个图有123个节点，先加载进来上一次训练的神经网络，第一层为新加入的123节点层，第二层为新加入的91节点层，后面依次加入上一个训练好的模型（除掉上一个模型的第一层），最后加入一个新的123层output，所以最后第二次的神经网络为[123,91(新构建的无参数训练过),88,66,44,22,44,66,88,91,123]。输入和输出为同一个123x123的邻接矩阵，一次训练一个的话就会有123个batches，第一次将整个邻接矩阵训练100个epochs.

3. 直到训练完最后一个图，取其前半部分encoder得到的22维embedding向量，重新输入同一个邻接矩阵，在embedding层停止得到输出，即为每个节点的22维embedding向量。


link_prediction.py
```python
import numpy as np
from sklearn import metrics # version=0.20.1
from sklearn.model_selection import train_test_split
import networkx as nx
from keras.models import Model, Sequential, load_model
import matplotlib.pyplot as plt
import sys


file_prefix = 'snapshots/s'
graph_file = file_prefix + str(7) + '_graph.gpickle'
g = nx.read_gpickle(graph_file)
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)
N = g.number_of_nodes()


final_model = load_model("models/prev_model_7.h5")

adj_mat = nx.adjacency_matrix(g).toarray()

# test_ratio = 0.50
# train_set, test_edges = train_test_split(g.edges(), test_size=test_ratio)
# g.remove_edges_from(test_edges)

# adj_mat = nx.adjacency_matrix(g).toarray()

reconstructed_adj = final_model.predict(adj_mat)

# reconstructed_adj = np.reshape(reconstructed_adj, (-1, 2))
# adj_mat = np.reshape(adj_mat, (-1, 2))

rows = adj_mat.shape[0]
cols = adj_mat.shape[1]
y_act = []

for x in range(0, rows):
    for y in range(0, cols):
        y_act.append(adj_mat[x,y])

pred =  []   
for x in range(0, rows):
    for y in range(0, cols):
        pred.append(reconstructed_adj[x,y])


fpr, tpr, thresholds = metrics.roc_curve(y_act, pred)
print(metrics.auc(fpr, tpr))
roc_auc = metrics.auc(fpr, tpr)


#Plotting

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```
对最后的模型同样还输入最后一个图的邻接矩阵，得到个预测的邻接矩阵，对比这两个邻接矩阵得到预测的AUC


[code reference](https://github.com/paulpjoby/DynGEM) and [dataset](https://github.com/paulpjoby/DynGEM/tree/master/datasets/haggle_snapshots/snapshots)