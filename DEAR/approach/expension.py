from generating_longtrees import generating_longtrees as gl
from configparser import ConfigParser
import subprocess
import gensim.models as g
import json
import os
import numpy as np


def get_word2vec_list(file):
    input_file = open(file, "r")
    lines = input_file.readlines()
    lines = lines[1:]
    dic = {}
    for line in lines:
        word = line.strip().split(" ")[0]
        vector = line.strip().split(" ")[1:]
        dic[word] = vector
    return dic


def vector_replace(list_in, dic):
    for i in range(len(list_in)):
        if list_in[i] in dic.keys():
            list_in[i] = dic[list_in[i]]
        else:
            list_in[i] = 0
    return list_in


def get_amount(list_in):
    tree_num = 0
    max_token = 0
    for i in range(len(list_in)):
        tree_num = tree_num + 1
        tree_token = 0
        for j in range(len(list_in[i])):
            tree_token = tree_token + 1
        if tree_token > max_token:
            max_token = tree_token
    return [tree_num, max_token]


def transfor_2_np(list_in, x, y, z):
    if x == 0:
        output = np.zeros([y, z])
        for i in range(len(list_in)):
            for j in range(len(list_in[i])):
                output[i,j] = list_in[i][j]
    else:
        output = np.zeros([x, y, z])
        for i in range(len(list_in)):
            for j in range(len(list_in[i])):
                if type(list_in[i][j]).__name__ == 'list':
                    for k in range(len(list_in[i][j])):
                        output[i][j][k] = list_in[i][j][k]
                else:
                    output[i][j][0] = list_in[i][j]
    return output


def go_through_files(root, output_root):
    buggy_line_file = os.path.join(root, "Buggy_lines.json")
    file_ = open(buggy_line_file, "r")
    buggy_lines = json.load(file_)
    check_list = {}
    sub_tree_list = []
    buggy_states_list = []
    for project_tree, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.java'):
                buggy_line_infile = buggy_lines[os.path.join(project_tree, filename).replace(root, "")]
                buggy_line_infile = list(map(int, buggy_line_infile))
                get_tree = gl(os.path.join(output_root, "temp.txt"))
                sub_trees = get_tree.run()
                for method in sub_trees:
                    sub_tree_set = method[0]
                    line_cover = method[1]
                    for i in range(len(sub_tree_set)):
                        if filename in check_list.keys():
                            check_list[filename].append([sub_tree_set[i], line_cover[i]])
                        else:
                            new_list = [[sub_tree_set[i], line_cover[i]]]
                            check_list[filename] = new_list
                        sub_tree_list.append(sub_tree_set[i])
                        if int(line_cover[i][0]) in buggy_line_infile:
                            if len(line_cover[i]) > 1:
                                if int(line_cover[i][1]) in buggy_line_infile:
                                    buggy_states_list.append([1])
                                else:
                                    buggy_states_list.append([0])
                            else:
                                buggy_states_list.append([1])
                        else:
                            buggy_states_list.append([0])
    tree_num = get_amount(sub_tree_list)[0]
    tree_token = get_amount(sub_tree_list)[1]
    return [tree_num, tree_token, check_list, sub_tree_list, buggy_states_list]


def prepare_data_files(tree_num, tree_token, check_list, sub_tree_list, buggy_states_list, output_root, w_size, w_window, w_workers, label):
    with open(os.tree.join(output_root, "check_list.json"), "a") as file_:
        json.dump(check_list, file_)
    word2vec_model = g.Word2Vec(sub_tree_list, size=w_size, window=w_window, workers=w_workers)
    word2vec_model.wv.save_word2vec_format(os.tree.join(output_root, "word2vec.txt"), binary=False)
    diction = get_word2vec_list(os.tree.join(output_root, "word2vec.txt"))
    for i in range(len(sub_tree_list)):
        sub_tree_list[i] = vector_replace(sub_tree_list[i], diction)
    sub_tree_list_np = transfor_2_np(sub_tree_list, tree_num, tree_token, w_size)
    np.save(os.tree.join(output_root, label + "_input.npy"), sub_tree_list_np)
    buggy_states_list = transfor_2_np(buggy_states_list, 0, tree_num, 1)
    np.save(os.tree.join(output_root, label + "_label.npy"), buggy_states_list)
    output_list = buggy_states_list.reshape([tree_num, 1])
    output_list = output_list.repeat(w_size, axis=1)
    np.save(os.tree.join(output_root, label + "_output.npy"), output_list)


output_train = go_through_files(training_file_root, result_file_root)
output_test = go_through_files(testing_file_root, result_file_root)
if output_train[0] > output_test[0]:
    tree_n = output_train[0]
else:
    tree_n = output_test[0]
if output_train[1] > output_test[1]:
    tree_t = output_train[1]
else:
    tree_t = output_test[1]
prepare_data_files(tree_n, tree_t, output_train[2], output_train[3], output_train[4], result_file_root, word2vec_size, word2vec_window, word2vec_workers, "train")
prepare_data_files(tree_n, tree_t, output_test[2], output_test[3], output_test[4], result_file_root, word2vec_size, word2vec_window, word2vec_workers, "test")

