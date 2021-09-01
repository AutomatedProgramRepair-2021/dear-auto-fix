import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling1D, Dropout
from keras.models import Input, Model
from keras.optimizers import adam
from TreeLSTM import treelstm
import numpy as np
import os


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


def grouping_for_method(local_v, dfg_v, pdg_v, method_list, input_length, input_dim):
    combine_1 = local_v*dfg_v
    combine_2 = local_v*pdg_v
    combine = (combine_1 + combine_2)*0.5
    method_num = get_method_amount(method_list)
    method_matrix = np.zeros([method_num, input_length, input_dim])
    method_id = -1
    count = 0
    for i in range(len(method_list)):
        if method_list[i] != method_id:
            method_id = method_list[i]
            count = 0
        for j in range(len(combine[i])):
            method_matrix[method_id][count][j] = combine[i][j]
        count = count + 1
    return method_matrix


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


input_l = get_method_max_length(method_l)
method_m = grouping_for_method(local_data, dfg_data, pdg_data, method_l, input_l, input_d)
model_local = global_context_learning(input_l, input_d, output_d, hidden_dim, learning_r, drop_r)
model_local.fit(method_m, output_data, batch_size=batch_size_num, epochs=epoch_num)
input_l = get_method_max_length(method_l)
method_m = grouping_for_method(local_data, dfg_data, pdg_data, method_l, input_l, input_d)
local_context_result = model_local.predict(method_m)
