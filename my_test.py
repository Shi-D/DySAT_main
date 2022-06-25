from __future__ import division
from __future__ import print_function

import json
import os
import time
from datetime import datetime

import logging
import scipy

from eval.link_prediction import evaluate_classifier, write_to_csv
from flags import *
from models.DySAT.models import DySAT
from utils.minibatch import *
from utils.preprocess import *
from utils.utilities import *



def construct_placeholders(num_time_steps):
    min_t = 0
    if FLAGS.window > 0:
        min_t = max(num_time_steps - FLAGS.window - 1, 0)
    placeholders = {
        'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, num_time_steps)],
        # [None,1] for each time step.
        'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, num_time_steps)],
        # [None,1] for each time step.
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),  # [None,1]
        'features': [tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, num_time_steps)],
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
                 range(min_t, num_time_steps)],
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop')
    }
    return placeholders



np.random.seed(123)
tf.set_random_seed(123)

flags = tf.app.flags
FLAGS = flags.FLAGS
# print('flags', FLAGS.log_dir)
# print('----------')

# Assumes a saved base model as input and model name to get the right directory.
output_dir = "./logs/{}_{}/".format(FLAGS.base_model, FLAGS.model)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# config_file = output_dir + "flags_{}.json".format(FLAGS.dataset)
#
# with open(config_file, 'r') as f:
#     config = json.load(f)
#     for name, value in config.items():
#         if name in FLAGS.__flags:
#             FLAGS.__flags[name].value = value

print("Updated flags", FLAGS.flag_values_dict().items())

# Set paths of sub-directories.
LOG_DIR = output_dir + FLAGS.log_dir
SAVE_DIR = output_dir + FLAGS.save_dir
CSV_DIR = output_dir + FLAGS.csv_dir
MODEL_DIR = output_dir + FLAGS.model_dir

path_emb = './output/DY-EPINIONS/emb_50/'

if not os.path.isdir(path_emb):
    os.makedirs(path_emb)

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)

if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()

# Setup logging
log_file = LOG_DIR + '/%s_%s_%s_%s_%s.log' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))

log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

logging.info(FLAGS.flag_values_dict().items())

# Create file name for result log csv from certain flag parameters.
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

# model_dir is not used in this code for saving.

# utils folder: utils.py, random_walk.py, minibatch.py
# models folder: layers.py, models.py
# main folder: train.py
# eval folder: link_prediction.py

"""
#1: Train logging format: Create a new log directory for each run (if log_dir is provided as input). 
Inside it,  a file named <>.log will be created for each time step. The default name of the directory is "log" and the 
contents of the <>.log will get appended per day => one log file per day.

#2: Model save format: The model is saved inside model_dir. 

#3: Output save format: Create a new output directory for each run (if save_dir name is provided) with embeddings at 
each 
time step. By default, a directory named "output" is created.

#4: Result logging format: A csv file will be created at csv_dir and the contents of the file will get over-written 
as per each day => new log file for each day.
"""

# Load graphs and features.

num_time_steps = FLAGS.time_steps
all_node_num = FLAGS.all_node_num

edge_lists, edge_ws, adjs, seed_sets, feats, gt_lists, graphs = load_data(FLAGS)
adjs = list(adjs)
adj_sparse = []
for adj in adjs:
    adj_sparse.append(sp.coo_matrix(adj).tocoo())

num_features = FLAGS.num_features

print('feats.shape', feats.shape)  # feats [50, 4039, 1]  50个不同的seeds


adj_train = []
feats_train = []
num_features_nonzero = []
loaded_pairs = False


# Load training context pairs (or compute them if necessary)
print('# training context pairs')
context_pairs_train = get_context_pairs(graphs, num_time_steps, FLAGS.seed_size, FLAGS.filepath)
print('# End training context pairs')

# Load evaluation data.
print('# Load evaluation data')
train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    get_evaluation_data(adj_sparse, num_time_steps, FLAGS.seed_size, FLAGS.filepath)
print('# End Load evaluation data')

# Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
# inductive testing.
new_G = nx.MultiGraph()
new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))

for e in graphs[num_time_steps - 2].edges():
    new_G.add_edge(e[0], e[1])

graphs[num_time_steps - 1] = new_G
adjs[num_time_steps - 1] = nx.adjacency_matrix(new_G)

print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))
logging.info("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

# Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
adj_train = list(map(lambda adj: normalize_graph_gcn(adj), adjs))


feats = np.reshape(feats, (-1, FLAGS.all_node_num, 1))  # feats [50, 4039, 1]  50个不同的seeds
num_features = feats[0].shape[1]
assert num_time_steps == len(adjs), 'num_time_steps != len(adjs)'  # So that, (t+1) can be predicted.

placeholders = construct_placeholders(num_time_steps)


check_file = "./model/"+str(FLAGS.dataset)+"/dysat_lr"+str(FLAGS.learning_rate)+"_size"+str(FLAGS.seed_size)+"_time"+str(FLAGS.time_steps)
# check_file = "./model/"+str(FLAGS.dataset)

# if not os.path.isdir(check_file):
#     os.makedirs(check_file)



feats_tmp = np.repeat(np.reshape(feats[0], (1, all_node_num, 1)), num_time_steps, axis=0)
feats_train = list(map(lambda feat: sparse_to_tuple(sp.coo_matrix(feat)), feats_tmp))
minibatchIterator = NodeMinibatchIterator(graphs, feats_train, adj_train,
                                          placeholders, num_time_steps, FLAGS, batch_size=FLAGS.batch_size,
                                          context_pairs=context_pairs_train)
model = DySAT(placeholders, num_features, num_features_nonzero, minibatchIterator.degs)


for s in range(0, FLAGS.seed_num):
    print("Initializing session")
    # Initialize session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print('-----------------------------------seed_no:', s)
    feats_tmp = np.repeat(np.reshape(feats[s], (1, all_node_num, 1)), num_time_steps, axis=0)
    # feats [51, 100, 1]

    print('feats_tmp.shape', feats_tmp.shape)
    feats_train = list(map(lambda feat: sparse_to_tuple(sp.coo_matrix(feat)), feats_tmp))
    num_features_nonzero = [x[1].shape[0] for x in feats_train]


    minibatchIterator = NodeMinibatchIterator(graphs, feats_train, adj_train,
                                              placeholders, num_time_steps, FLAGS, batch_size=FLAGS.batch_size,
                                              context_pairs=context_pairs_train)


    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, check_file)
    # saver = tf.train.import_meta_graph("Model/model.ckpt.meta")

    # Result accumulator variables.
    epochs_test_result = defaultdict(lambda: [])
    epochs_val_result = defaultdict(lambda: [])
    epochs_embeds = []
    epochs_attn_wts_all = []

    for epoch in range(1):  # 每个种子集都有一个epoch
        minibatchIterator.shuffle()
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({placeholders['spatial_drop']: FLAGS.spatial_drop})
        feed_dict.update({placeholders['temporal_drop']: FLAGS.temporal_drop})

        minibatchIterator.test_reset()
        feed_dict.update({placeholders['spatial_drop']: 0.0})
        feed_dict.update({placeholders['temporal_drop']: 0.0})
        if FLAGS.window < 0:
            assert FLAGS.time_steps == model.final_output_embeddings.get_shape()[1]
        # emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:, model.final_output_embeddings.get_shape()[1] - 2, :]  # (100, 4, 128)
        emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:, -1, :]  # (100, 4, 128)
        emb = np.array(emb)  # (100, 128)
        # Use external classifier to get validation and test results.
        val_results, test_results, _, _ = evaluate_classifier(train_edges,
                                                              train_edges_false, val_edges, val_edges_false, test_edges,
                                                              test_edges_false, emb, emb)

        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]

        print("Epoch {}, Val AUC {}".format(epoch, epoch_auc_val))
        print("Epoch {}, Test AUC {}".format(epoch, epoch_auc_test))
        logging.info("Val results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_val))
        logging.info("Test results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_test))

        epochs_test_result["HAD"].append(epoch_auc_test)
        epochs_val_result["HAD"].append(epoch_auc_val)
        epochs_embeds.append(emb)


    print(len(emb))  # 100
    print(emb.shape)  # (100, 128)
    # np.savez(path_emb + 'output_dyfb_'+str(s) +'.npz', data=emb)
    np.savez('{}emb_dyepinions_{}.npz'.format(path_emb, str(s)), data=emb)
    # np.savez(SAVE_DIR + '/{}_embs_{}_{}.npz'.format(FLAGS.model, FLAGS.dataset, FLAGS.time_steps), data=emb)

    # sess.close()
    # tf.reset_default_graph()

