import numpy as np
import scipy.sparse as sp
import tensorflow as tf

tf = tf.compat.v1

"""
Disclaimer: functions defined from lines 18 to 54 in this file come from
the tkipf/gae original repository on Graph Autoencoders. Moreover, the
mask_test_edges function is borrowed from philipjackson's mask_test_edges 
pull request on this same repository.
"""


def sparse_to_tuple(sparse_mx):

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape


def preprocess_graph(adj):

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])

    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())

    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj_normalized_layer2, adj_orig, features, deg_matrix, placeholders,simple):

    # Construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_layer2']: adj_normalized_layer2})
    feed_dict.update({placeholders['adj_orig']: adj_orig})

    if not simple:
        feed_dict.update({placeholders['degree_matrix']: deg_matrix})

    return feed_dict

def preprocess_degree(adj, is_simple):

    """
    Preprocessing degree-based term for modularity loss
    :param adj: sparse adjacency matrix of the graph
    :param is_simple: "simple" boolean flag for modularity
    :return: degree-based term matrices
    """

    if is_simple:
        deg_matrix = None
        deg_matrix_init = None

    else:
        deg = np.sum(adj, 1)
        deg_matrix = (1.0 / np.sum(adj)) * deg.dot(np.transpose(deg))
        #deg_matrix = deg_matrix - np.diag(np.diag(deg_matrix))
        deg_matrix_init = sp.csr_matrix(deg_matrix)
        deg_matrix = sparse_to_tuple(deg_matrix_init)

    return deg_matrix, deg_matrix_init

