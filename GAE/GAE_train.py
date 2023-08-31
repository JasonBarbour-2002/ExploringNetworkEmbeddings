import numpy as np
import scipy.sparse as sp
import networkx as nx
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from GAE.louvain import louvain_clustering
from GAE.model import  GCNModelAE, GCNModelVAE, LinearModelAE, LinearModelVAE
from GAE.optimizer import OptimizerAE, OptimizerVAE
from GAE.preprocessing import *
from GAE.sampling import get_distribution, node_sampling
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf = tf.compat.v1
tf.disable_eager_execution()
class GAE:
    def __init__(
        self,
        dimensions: int = 128,
        features:str = None,
        model: str = 'linear_vae',
        dropout: int = 0,
        iterations: int = 200,
        learning_rate: float = 0.01,
        hidden: int = 32,
        beta: float = 0.0,
        lamb: float = 0.0,
        gamma: float = 1.0,
        s_reg: int = 2,
        fast_gae: bool = False,
        nb_node_samples: int = 1000,
        measure: str = 'degree',
        alpha: float = 1.0,
        replace: bool = False,
        simpler: bool = True,
    ):
        self.dimensions = dimensions
        self.features = features
        self.model = model
        self.dropout = dropout
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.hidden = hidden
        self.beta = beta
        self.lamb = lamb
        self.gamma = gamma
        self.s_reg = s_reg
        self.fast_gae = fast_gae
        self.nb_node_samples = nb_node_samples
        self.measure = measure
        self.alpha = alpha
        self.replace = replace
        self.simpler = simpler

    
    def fit(self,Graph):
        adj = nx.to_scipy_sparse_array(Graph)
        num_nodes = adj.shape[0]
        if self.features is None:
            features = sp.identity(num_nodes)
        else:
            features = self.features
        features = sparse_to_tuple(features)
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]
        adj_louvain_init, nb_communities_louvain, partition = louvain_clustering(adj, self.s_reg)
        if self.fast_gae:
            node_distribution = get_distribution(self.measure, self.alpha, adj)
            # Node sampling for initializations
            sampled_nodes, adj_label, adj_sampled_sparse = node_sampling(adj, node_distribution,self.nb_node_samples, self.replace)
        else:
            sampled_nodes = np.array(range(self.nb_node_samples))

        placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_layer2': tf.sparse_placeholder(tf.float32), # Only used for 2-layer GCN encoders
        'degree_matrix': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape = ()),
        'sampled_nodes': tf.placeholder_with_default(sampled_nodes, shape = [self.nb_node_samples])
        }
        # Create model
        if self.model == 'linear_ae':
            # Linear Graph Autoencoder
            model = LinearModelAE(placeholders, num_features, features_nonzero,dimension= self.dimensions,fastgae = self.fast_gae)
        elif self.model == 'linear_vae':
            # Linear Graph Variational Autoencoder
            model = LinearModelVAE(placeholders, num_features, num_nodes, features_nonzero,dimension= self.dimensions,fastgae = self.fast_gae)
        elif self.model == 'gcn_ae':
            # 2-layer GCN Graph Autoencoder
            model = GCNModelAE(placeholders, num_features, features_nonzero,dimension= self.dimensions,fastgae = self.fast_gae,hidden = self.hidden)
        elif self.model == 'gcn_vae':
            # 2-layer GCN Graph Variational Autoencoder
            model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero,dimension= self.dimensions,fastgae = self.fast_gae,hidden = self.hidden)
        else:
            raise ValueError('Undefined model!')
    
        if self.fast_gae:
            num_sampled = adj_sampled_sparse.shape[0]
            sum_sampled = adj_sampled_sparse.sum()
            Temp = num_sampled * num_sampled 
            pos_weight = float(Temp - sum_sampled) / sum_sampled
            norm = Temp / float((Temp- sum_sampled) * 2)
        else:
            Temp= num_nodes * num_nodes
            pos_weight = float(Temp - adj.sum()) / adj.sum()
            norm = Temp / float((Temp- adj.sum()) * 2)
        
        l = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],validate_indices = False), [-1])
        d = tf.reshape(tf.sparse_tensor_to_dense(placeholders['degree_matrix'],validate_indices = False), [-1])
        if self.model in ('gcn_ae', 'linear_ae'):
            opt = OptimizerAE(
                        preds = model.reconstructions,
                        labels = l,
                        degree_matrix = d,
                        num_nodes = num_nodes,
                        pos_weight = pos_weight,
                        norm = norm,
                        clusters_distance = model.clusters,
                        simple= self.simpler,
                        fastgae = self.fast_gae,
                        learning_rate = self.learning_rate,
                        beta=  self.beta,)
        else:
            opt = OptimizerVAE(
                        preds = model.reconstructions,
                        labels = l,
                        degree_matrix = d,
                        model = model,
                        num_nodes = num_nodes,
                        pos_weight = pos_weight,
                        norm = norm,
                        clusters_distance = model.clusters,
                        simple= self.simpler,
                        fastgae= self.fast_gae,
                        learning_rate= self.learning_rate,
                        beta=  self.beta,)
            
        adj_norm = preprocess_graph(adj + self.lamb*adj_louvain_init)
        adj_norm_layer2 = preprocess_graph(adj)
        if not self.fast_gae:
            adj_label = sparse_to_tuple(adj + sp.eye(num_nodes))
            
        deg_matrix, deg_matrix_init = preprocess_degree(adj, self.simpler)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        
        for i in range(self.iterations):
            feed_dict = construct_feed_dict(adj_norm, adj_norm_layer2, adj_label, \
                                        features, deg_matrix, placeholders,self.simpler)
            if self.fast_gae:
                # Update sampled subgraph and sampled Louvain matrix
                feed_dict.update({placeholders['sampled_nodes']: sampled_nodes})
                # New node sampling
                sampled_nodes, adj_label, adj_label_sparse, = node_sampling(adj, \
                                                                            node_distribution, \
                                                                            self.nb_node_samples, \
                                                                            self.replace)

            # Weights update
            outs = sess.run([opt.opt_op, opt.cost, opt.cost_adj, opt.cost_mod],feed_dict = feed_dict)
            
        self.emb = sess.run(model.z_mean, feed_dict = feed_dict)
        
    def get_embedding(self):
        return self.emb