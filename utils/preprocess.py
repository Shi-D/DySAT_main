from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
from utilities import run_random_walks_n2v
import dill

flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(123)


def load_data(FLAGS):
    path = FLAGS.filepath
    data_name = FLAGS.dataset
    g_num = FLAGS.graph_num
    seed_size = FLAGS.seed_size
    all_node_num = FLAGS.all_node_num


    edge_lists=[]
    edge_ws=[]
    adjs=[]
    seed_sets=[]
    feats = []
    gt_lists=[]
    g_lists=[]

    for g_i in range(g_num):
        # 读取边
        file_handler = open(path+data_name+'.'+str(g_i)+'.edges', mode='r')
        edge_content = file_handler.readlines()
        edge_list = []
        for edge in edge_content:
            edge = edge.strip('\n')
            edge = edge.split(' ')
            edge_list.append([int(edge[0]), int(edge[1])])

        # 读取边权重
        file_handler = open(path+data_name+'.'+str(g_i)+'.weight', mode='r')
        edge_content = file_handler.readlines()
        edge_w = []
        for w in edge_content:
            w = w.strip('\n')
            edge_w.append(float(w))


        # 获取图的邻接矩阵
        g = nx.DiGraph()
        g.add_nodes_from(range(all_node_num))
        g.add_weighted_edges_from(np.c_[np.array(edge_list), np.array(edge_w).squeeze()],
                                  weight='weight')

        nodes = sorted(g.nodes())
        adj = nx.adjacency_matrix(g, nodes).todense()
        adj = adj + np.eye(len(nodes))


        edge_lists.append(edge_list)
        edge_ws.append(edge_w)
        adjs.append(adj)
        g_lists.append(g)

    # 读取种子节点集
    file_handler = open(path+data_name+'.'+str(seed_size)+'.seeds', mode='r')
    seed_content = file_handler.readlines()
    for s in seed_content:
        s = s.strip('\n')
        s = s.strip(' ')
        s = s.split(' ')
        s = np.array(s).astype(int)
        tmp = [0 for _ in range(all_node_num)]
        tmp = np.array(tmp)
        tmp[s] = 1
        feats.append(tmp)
        seed_sets.append(list(s))


    # 读取 ground-truth
    file_handler = open(path+data_name+'.'+str(seed_size)+'.groundtruth', mode='r')
    gt_content = file_handler.readlines()
    for gt in gt_content:
        gt = gt.strip('\n')
        gt = gt.strip(' ')
        gt = gt.split(' ')
        gt = np.array(gt).astype(float)
        gt = list(gt)
        gt_lists.append(gt)

    return edge_lists, edge_ws, adjs, seed_sets, np.array(feats), gt_lists, g_lists


def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
    graphs = np.load("data/{}/{}".format(dataset_str, "graphs.npz"),  encoding='bytes', allow_pickle=True)['graph']
    print("Loaded {} graphs ".format(len(graphs)))
    print(type(graphs[0]))
    adj_matrices = map(lambda x: nx.adjacency_matrix(x), graphs)
    return graphs, adj_matrices


def load_feats(dataset_str):
    """ Load node attribute snapshots given the name of dataset (not used in experiments)"""
    features = np.load("data/{}/{}".format(dataset_str, "features.npz"), allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features


def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     features = r_mat_inv.dot(features)
#     return features.todense(), sparse_to_tuple(features)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sp.coo_matrix(adj)
    # adj_ = adj + sp.eye(adj.shape[0])
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def get_context_pairs_incremental(graph):
    return run_random_walks_n2v(graph, graph.nodes())


def get_context_pairs(graphs, num_time_steps, seed_size, filepath):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    load_path = "{}train_pairs_n2v_{}.pkl".format(filepath, str(seed_size))
    try:
        context_pairs_train = dill.load(open(load_path, 'rb'))
        print("Loaded context pairs from pkl file directly")
    except (IOError, EOFError):
        print("Computing training pairs ...")
        context_pairs_train = []
        for i in range(0, num_time_steps):
            context_pairs_train.append(run_random_walks_n2v(graphs[i], graphs[i].nodes()))
        dill.dump(context_pairs_train, open(load_path, 'wb'))
        print("Saved pairs")

    return context_pairs_train


def get_evaluation_data(adjs, num_time_steps, seed_size, filepath):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = num_time_steps - 2
    eval_path = "{}/eval_{}.npz".format(filepath, str(seed_size))
    try:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
        print("Loaded eval data")
    except IOError:
        next_adjs = adjs[eval_idx + 1]
        print("Generating and saving eval data ....")
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            create_data_splits(adjs[eval_idx], next_adjs, val_mask_fraction=0.2, test_mask_fraction=0.6)
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                           test_edges, test_edges_false]))
        print("End Generating and saving eval data ....")

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false

