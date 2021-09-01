import math
import tensorflow as tf
from utils import reduce_sum
from utils import softmax
from utils import get_shape
import numpy as np

stddev = 0.01
batch_size = 1
iter_routing = 3

def squash(vector):
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

def dynamic_routing(shape, input, num_outputs=10, num_dims=16):
    """The Dynamic Routing Algorithm proposed by Sabour et al."""
    
    input_shape = shape
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))
    
    delta_IJ = tf.zeros([input_shape[0], input_shape[1], num_outputs, 1, 1], dtype=tf.dtypes.float32)

    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])

    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            gamma_IJ = softmax(delta_IJ, axis=2)

            if r_iter == iter_routing - 1:
                s_J = tf.multiply(gamma_IJ, u_hat)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(gamma_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                delta_IJ += u_produce_v

    return(v_J)

def vts_routing(input, top_a, top_b, num_outputs, num_dims):
    """The proposed Variable-to-Static Routing Algorithm."""

    alpha_IJ = tf.zeros((int(num_outputs/top_a*top_b), num_outputs), dtype=tf.float32)
    input = tf.transpose(input,perm=[0,2,3,1])
    u_i,_ = tf.nn.top_k(input,k=top_b)
    u_i = tf.transpose(u_i,perm=[0,3,1,2])
    u_i = tf.reshape(u_i, (-1, num_dims))
    u_i = tf.stop_gradient(u_i)
    
    input,_ = tf.nn.top_k(input,k=top_a)
    input = tf.transpose(input,perm=[0,3,1,2])
    v_J = input
    v_J = tf.reshape(v_J, (-1, num_dims))
        
    for rout in range(1):
        u_produce_v = tf.matmul(u_i, v_J,transpose_b=True)
        alpha_IJ += u_produce_v
        beta_IJ = softmax(alpha_IJ,axis=-1)
        v_J = tf.matmul(beta_IJ,u_i,transpose_a=True)
    
    v_J = tf.reshape(v_J,(1, num_outputs, num_dims, 1))
    return squash(v_J)

def init_net_treecaps(feature_size, label_size):
    """Initialize an empty TreeCaps network."""
    top_a = 20
    top_b = 25
    num_conv = 8
    output_size = 128
    caps1_num_dims = 8
    caps1_num_caps = int(num_conv*output_size/caps1_num_dims)*top_a
    caps1_out_caps = label_size
    caps1_out_dims = 8

    with tf.name_scope('inputs'):
        nodes = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
        children = tf.placeholder(tf.int32, shape=(None, None, None), name='children')

    with tf.name_scope('network'):  
        """The Primary Variable Capsule Layer."""
        primary_variable_caps = primary_variable_capsule_layer(num_conv, output_size, nodes, children, feature_size, caps1_num_dims)
        
        """The Primary Static Capsule Layer."""
        primary_static_caps = vts_routing(primary_variable_caps,top_a,top_b,caps1_num_caps,caps1_num_dims)        
        primary_static_caps = tf.reshape(primary_static_caps, shape=(batch_size, -1, 1, caps1_num_dims, 1))
        
        """The Code Capsule Layer."""
        #Get the input shape to the dynamic routing algorithm
        dr_shape = [batch_size,caps1_num_caps,1,caps1_num_dims,1]
        codeCaps = dynamic_routing(dr_shape, primary_static_caps, num_outputs=caps1_out_caps, num_dims=caps1_out_dims)
        codeCaps = tf.squeeze(codeCaps, axis=1)
        
        """Obtaining the classification output."""
        v_length = tf.sqrt(reduce_sum(tf.square(codeCaps),axis=2, keepdims=True) + 1e-9)
        out = tf.reshape(v_length,(-1,label_size))

    return nodes, children, out


def primary_variable_capsule_layer(num_conv, output_size, nodes, children, feature_size, num_dims):
    """The proposed Primary Variable Capsule Layer."""
    
    with tf.name_scope('primary_variable_capsule_layer'):
        nodes = [
            tf.expand_dims(conv_node(nodes, children, feature_size, output_size),axis=-1)
            for _ in range(num_conv)
        ]    
        conv_output = tf.concat(nodes, axis=-1)
        primary_variable_capsules = tf.reshape(conv_output,shape=(1,-1,output_size,num_conv))
        return primary_variable_capsules

def conv_layer(num_conv, output_size, nodes, children, feature_size):
    """Creates a convolution layer with num_conv convolutions merged together at
    the output. Final output will be a tensor with shape
    [batch_size, num_nodes, output_size * num_conv]"""

    with tf.name_scope('conv_layer'):
        nodes = [
            tf.expand_dims(conv_node(nodes, children, feature_size, output_size),axis=-1)
            for _ in range(num_conv)
        ]     
        return tf.concat(nodes, axis=-1)

def conv_node(nodes, children, feature_size, output_size):
    """Perform convolutions over every batch sample."""
    with tf.name_scope('conv_node'):
        std = 1.0 / math.sqrt(feature_size)
        w_t, w_l, w_r = (
            tf.Variable(tf.random.truncated_normal([feature_size, output_size], stddev=std), name='Wt'),
            tf.Variable(tf.random.truncated_normal([feature_size, output_size], stddev=std), name='Wl'),
            tf.Variable(tf.random.truncated_normal([feature_size, output_size], stddev=std), name='Wr'),
        )
        init = tf.random.truncated_normal([output_size,], stddev=math.sqrt(2.0/feature_size))

        b_conv = tf.Variable(init, name='b_conv')

        return conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv)

def conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv):
    """Convolve a batch of nodes and children.
    Lots of high dimensional tensors in this function. Intuitively it makes
    more sense if we did this work with while loops, but computationally this
    is more efficient. Don't try to wrap your head around all the tensor dot
    products, just follow the trail of dimensions.
    """
    with tf.name_scope('conv_step'):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        # children is shape (batch_size x max_tree_size x max_children)

        with tf.name_scope('trees'):
            # children_vectors will have shape
            # (batch_size x max_tree_size x max_children x feature_size)
            children_vectors = children_tensor(nodes, children, feature_size)

            # add a 4th dimension to the nodes tensor
            nodes = tf.expand_dims(nodes, axis=2)
            # tree_tensor is shape
            # (batch_size x max_tree_size x max_children + 1 x feature_size)
            tree_tensor = tf.concat([nodes, children_vectors], axis=2, name='trees')

        with tf.name_scope('coefficients'):
            # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
            c_t = eta_t(children)
            c_r = eta_r(children, c_t)
            c_l = eta_l(children, c_t, c_r)

            # concatenate the position coefficients into a tensor
            # (batch_size x max_tree_size x max_children + 1 x 3)
            coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

        with tf.name_scope('weights'):
            # stack weight matrices on top to make a weight tensor
            # (3, feature_size, output_size)
            weights = tf.stack([w_t, w_r, w_l], axis=0)

        with tf.name_scope('combine'):
            batch_size = tf.shape(children)[0]
            max_tree_size = tf.shape(children)[1]
            max_children = tf.shape(children)[2]

            # reshape for matrix multiplication
            x = batch_size * max_tree_size
            y = max_children + 1
            result = tf.reshape(tree_tensor, (x, y, feature_size))
            coef = tf.reshape(coef, (x, y, 3))
            result = tf.matmul(result, coef, transpose_a=True)
            result = tf.reshape(result, (batch_size, max_tree_size, 3, feature_size))

            # output is (batch_size, max_tree_size, output_size)
            result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

            # output is (batch_size, max_tree_size, output_size)
            return tf.nn.tanh(result + b_conv)



def children_tensor(nodes, children, feature_size):
    """Build the children tensor from the input nodes and child lookup."""
    with tf.name_scope('children_tensor'):
        max_children = tf.shape(children)[2]
        batch_size = tf.shape(nodes)[0]
        num_nodes = tf.shape(nodes)[1]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, feature_size))
        # vector_lookup is (batch_size x num_nodes x feature_size)
        vector_lookup = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children = tf.expand_dims(children, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children = tf.concat([batch_indices, children], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x feature_size)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children, name='children')

def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    with tf.name_scope('coef_t'):
        # children is shape (batch_size x max_tree_size x max_children)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]
        # eta_t is shape (batch_size x max_tree_size x max_children + 1)
        return tf.tile(tf.expand_dims(tf.concat(
            [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
            axis=1), axis=0,
        ), [batch_size, 1, 1], name='coef_t')

def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belogs to the 'right'"""
    with tf.name_scope('coef_r'):
        # children is shape (batch_size x max_tree_size x max_children)
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]

        # num_siblings is shape (batch_size x max_tree_size x 1)
        num_siblings = tf.cast(
            tf.count_nonzero(children, axis=2, keep_dims=True),
            dtype=tf.float32
        )
        # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
        num_siblings = tf.tile(
            num_siblings, [1, 1, max_children + 1], name='num_siblings'
        )
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2, name='mask'
        )

        # child indices for every tree (batch_size x max_tree_size x max_children + 1)
        child_indices = tf.multiply(tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                    axis=0
                ),
                axis=0
            ),
            [batch_size, max_tree_size, 1]
        ), mask, name='child_indices')

        # weights for every tree node in the case that num_siblings = 0
        # shape is (batch_size x max_tree_size x max_children + 1)
        singles = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.fill((batch_size, max_tree_size, 1), 0.5),
             tf.zeros((batch_size, max_tree_size, max_children - 1))],
            axis=2, name='singles')

        # eta_r is shape (batch_size x max_tree_size x max_children + 1)
        return tf.where(
            tf.equal(num_siblings, 1.0),
            # avoid division by 0 when num_siblings == 1
            singles,
            # the normal case where num_siblings != 1
            tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
            name='coef_r'
        )

def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    with tf.name_scope('coef_l'):
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
                tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2,
            name='mask'
        )

        # eta_l is shape (batch_size x max_tree_size x max_children + 1)
        return tf.multiply(
            tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
        )

def pooling_layer(nodes):
    """Creates a max dynamic pooling layer from the nodes."""
    with tf.name_scope("pooling"):
        pooled = tf.reduce_max(nodes, axis=1)
        return pooled


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def hidden_layer(pooled, input_size, output_size):
    """Create a hidden feedforward layer."""
    with tf.name_scope("hidden"):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_size, output_size], stddev=1.0 / math.sqrt(input_size)
            ),
            name='weights'
        )

        init = tf.truncated_normal([output_size,], stddev=math.sqrt(2.0/input_size))
        #init = tf.zeros([output_size,])
        biases = tf.Variable(init, name='biases')

        return lrelu(tf.matmul(pooled, weights) + biases, 0.01)


def loss_layer(logits_node, label_size):
    """Create a loss layer for training."""

    labels = tf.placeholder(tf.float32, (None, label_size,))

    with tf.name_scope('loss_layer'):
        max_l = tf.square(tf.maximum(0., 0.9 - logits_node))
        max_r = tf.square(tf.maximum(0., logits_node - 0.1))
        T_c = labels
        L_c = T_c * max_l + 0.5 * (1 - T_c) * max_r
        
        loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        return labels, loss

def out_layer(logits_node):
    """Apply softmax to the output layer."""
    with tf.name_scope('output'):
        return tf.nn.softmax(logits_node)