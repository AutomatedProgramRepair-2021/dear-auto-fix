import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling1D, Dropout
from keras.models import Input, Model
from keras.optimizers import adam
import sys
import preprocess
sys.path.append("..")
from tf_tree_lstm import tf_tree_lstm as treelstm
import numpy as np
import os
from evaluation import evaluation


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
ktf.set_session(session)


def get_method_max_length(method_list):
    max_length = 0
    length_record = 0
    method_id = -1
    for i in range(len(method_list)):
        if method_list[i] != method_id:
            method_id = method_list[i]
            if length_record > max_length:
                max_length = length_record
            length_record = 1
        else:
            length_record = length_record + 1
            if i != len(method_list) - 1 and length_record > max_length:
                max_length = length_record
    return max_length


def get_method_amount(method_list):
    method_num = 0
    method_id = -1
    for i in range(len(method_list)):
        if method_list[i] != method_id:
            method_id = method_list[i]
            method_num = method_num + 1
    return method_num


def tree_based_learning_model(input_length, input_dim, output_dim, hidden_dim, learning_rate, drop_rate):
    tree_input = Input(shape=(input_length, input_dim))
    structure_input = Input(shape=(input_length, input_dim))
    tree_output = treelstm(hidden_dim)([tree_input, structure_input])
    tree_output = treelstm(hidden_dim)(tree_output)
    output_fix = Dropout(rate=drop_rate)(tree_output)
    fixed_output = Dense(output_dim)(output_fix)
    stand_output = Activation(activation='softmax')(fixed_output)
    model = Model([tree_input, structure_input], stand_output)
    ada = adam(lr=learning_rate)
    model.compile(optimizer=ada, loss='categorical_crossentropy')
    return model


def getting_data(file_path, print_text):
    print("Loading " + print_text + " Data...")
    data = np.load(file_path)
    print("Done")
    return data

def run_model(dataset_, pre):
    dataset = getting_data(dataset_, dataset_)
    input_dim = get_method_max_length(dataset)
    model = tree_based_learning_model(get_method_amount (dataset),input_dim, input_dim, input_dim, 0.01, 0.5)
    if os.path.exists(path + "/models/autofix.5h"):
        model.load_model(path + "/models/autofix.5h")
    else:
        model.train([dataset[0], dataset[1]], dataset[2])
    output = model.predict(pre)
    output = preprocess.back_to_token(output)
    evaluation(output, dataset_)