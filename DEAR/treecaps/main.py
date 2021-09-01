from tensorflow import saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import os
import logging
import pickle
import tensorflow as tf
import numpy as np
import network as network
import sampling as sampling
import sys
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_program_data
from data_loader import MonoLanguageProgramData
import argparse
import random
import shutil
import progressbar
from keras_radam.training import RAdamOptimizer

parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=1, help='train batch size, always 1')
parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size, always 1')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--n_classes', type=int, default=10, help='manual seed')
parser.add_argument('--train_directory', default="Dataset_B/train", help='train program data') 
parser.add_argument('--test_directory', default="Dataset_B/test", help='test program data')
parser.add_argument('--model_path', default="Result_B_treecaps", help='path to save the model')
parser.add_argument('--training', action="store_true",help='is training')
parser.add_argument('--testing', action="store_true",help='is testing')
parser.add_argument('--training_percentage', type=float, default=1.0 ,help='percentage of data use for training')
parser.add_argument('--log_path', default="" ,help='log path for tensorboard')
parser.add_argument('--epoch', type=int, default=0, help='epoch to test')
parser.add_argument('--embeddings_directory', default="embedding/fast_pretrained_vectors.pkl")
parser.add_argument('--cuda', default="0",type=str, help='enables cuda')

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

if not os.path.isdir("cached"):
    os.mkdir("cached")


def train_model(train_trees, val_trees, labels, embeddings, embedding_lookup, opt):
    max_acc = 0.0
    logdir = opt.model_path
    batch_size = opt.train_batch_size
    epochs = opt.niter
    num_feats = len(embeddings[0])
    
    random.shuffle(train_trees)
    
    nodes_node, children_node, codecaps_node = network.init_net_treecaps(num_feats,len(labels))

    codecaps_node = tf.identity(codecaps_node, name="codecaps_node")

    out_node = network.out_layer(codecaps_node)
    labels_node, loss_node = network.loss_layer(codecaps_node, len(labels))

    optimizer = RAdamOptimizer(opt.lr)
    train_step = optimizer.minimize(loss_node)
    
     ### init the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))

    checkfile = os.path.join(logdir, 'tree_network.ckpt')

    print("Begin training..........")
    num_batches = len(train_trees) // batch_size + (1 if len(train_trees) % batch_size != 0 else 0)
    for epoch in range(1, epochs+1):
        bar = progressbar.ProgressBar(maxval=len(train_trees), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i, batch in enumerate(sampling.batch_samples(
            sampling.gen_samples(train_trees, labels, embeddings, embedding_lookup), batch_size
        )):
            nodes, children, batch_labels = batch
            step = (epoch - 1) * num_batches + i * batch_size

            if not nodes:
                continue
            _, err, out = sess.run(
                [train_step, loss_node, out_node],
                feed_dict={
                    nodes_node: nodes,
                    children_node: children,
                    labels_node: batch_labels
                }
            )
            bar.update(i+1)
        bar.finish()

        correct_labels = []
        predictions = []
        logits = []
        for batch in sampling.batch_samples(
            sampling.gen_samples(val_trees, labels, embeddings, embedding_lookup), 1
        ):
            nodes, children, batch_labels = batch
            output = sess.run([out_node],
                feed_dict={
                    nodes_node: nodes,
                    children_node: children
                }
            )
            correct_labels.append(np.argmax(batch_labels))
            predictions.append(np.argmax(output))
            logits.append(output)

        target_names = list(labels)
        acc = accuracy_score(correct_labels, predictions)
        if (acc>max_acc):
            max_acc = acc
            saver.save(sess, checkfile)
            np.save(opt.model_path+'/logits',np.array(logits))
            np.save(opt.model_path+'/correct',np.array(correct_labels))

        print('Epoch',str(epoch),'Accuracy:', acc, 'Max Acc: ',max_acc)
        csv_log.write(str(epoch)+','+str(acc)+','+str(max_acc)+'\n')

    print("Finish all iters, storring the whole model..........")


def test_model(test_trees, labels, embeddings, embedding_lookup, opt):
    
    logdir = opt.model_path
    batch_size = opt.train_batch_size
    epochs = opt.niter
    num_feats = len(embeddings[0])

    random.shuffle(test_trees)

    # build the inputs and outputs of the network
    nodes_node, children_node, codecaps_node = network.init_net_treecaps(num_feats,len(labels))
 
    out_node = network.out_layer(codecaps_node)
    labels_node, loss_node = network.loss_layer(codecaps_node, len(labels))

    optimizer = RAdamOptimizer(opt.lr)
    train_step = optimizer.minimize(loss_node)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))
                
    checkfile = os.path.join(logdir, 'tree_network.ckpt')

    correct_labels = []
    predictions = []
    print('Computing training accuracy...')
    for batch in sampling.batch_samples(
        sampling.gen_samples(test_trees, labels, embeddings, embedding_lookup), 1
    ):
        nodes, children, batch_labels = batch
        output = sess.run([out_node],
            feed_dict={
                nodes_node: nodes,
                children_node: children,
            }
        )
        correct_labels.append(np.argmax(batch_labels))
        predictions.append(np.argmax(output))

    target_names = list(labels)
    print(classification_report(correct_labels, predictions, target_names=target_names))
    print(confusion_matrix(correct_labels, predictions))
    print('*'*50)
    print('Accuracy:', accuracy_score(correct_labels, predictions))
    print('*'*50)


def main(opt):
    
    print("Loading embeddings....")
    with open(opt.embeddings_directory, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh,encoding='latin1')
       
    labels = [str(i) for i in range(1, opt.n_classes+1)]

    if opt.training:
        print("Loading train trees...")
        train_data_loader = MonoLanguageProgramData(opt.train_directory, 0, opt.n_classes)
        train_trees, _ = train_data_loader.trees, train_data_loader.labels

        val_data_loader = MonoLanguageProgramData(opt.test_directory, 2, opt.n_classes)
        val_trees, _ = val_data_loader.trees, val_data_loader.labels

        train_model(train_trees, val_trees,  labels, embeddings, embed_lookup , opt) 

    if opt.testing:
        print("Loading test trees...")
        test_data_loader = MonoLanguageProgramData(opt.test_directory, 1, opt.n_classes)
        test_trees, _ = test_data_loader.trees, test_data_loader.labels
        print("All testing trees : " + str(len(test_trees)))
        test_model(test_trees, labels, embeddings, embed_lookup , opt) 

if __name__ == "__main__":
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    csv_log = open(opt.model_path+'/log.csv', "w")
    csv_log.write('Epoch,Training Loss,Validation Accuracy\n')
    main(opt)
    csv_log.close()