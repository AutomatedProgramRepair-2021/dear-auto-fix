#from generating_longtrees import generating_longtrees as gl
import gensim.models as g
import json
from keras.backend import ones_like
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda
from keras.optimizers import adam
from keras.layers import Conv1D, Dense, Reshape, Concatenate, Flatten, Activation, Dropout, Softmax
from keras.models import Input, Model, load_model
from keras_self_attention import SeqSelfAttention
#from keras_multi_head import MultiHeadAttention
#from adding_weight import adding_weight
import numpy as np
import os, re
from gensim.models import Word2Vec


def get_glove_list(file):
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


def run_expansion():
    projects = ["Chart", "Closure", "Lang", "Math", "Mockito", "Time"]  
    bugs = [26, 133, 65, 106, 38, 27]
    path= os.getcwd()
    data_list = []
    data_label = []
    for i in range(44154):
        bug_id = i + 1
        source_file_root = path + "/data/CPatMiner/" + str(bug_id) + "/"
        label_file = open(source_file_root + "removed_lines.txt")
        label_lines = label_file.readlines()
        for line in label_lines:
            line = line.split(",")
            file_name = line[0]
            file_input = open(source_file_root + "/before/" + file_name,"r")
            input_lines = file_input.readlines()
            list_ = []
            label = []
            for i in range(11):
                list_.append(input_lines[int(line[1]) -6 + i])
                for line_ in label_lines:
                    line_ = line_.split(",")
                    if line_[0] == line[0] and ine(line_[1]) == int(line[1]) - 5 + i:
                        label.append(1)
                        break
                label.append(0)
            data_list.append(list_)
            data_label.append(label)
    input_data_ = np.array(data_list)
    output_data_ = np.array(data_lable)
    embedding_data = ""
    input_data = input_data_
    output_data = output_data_
    max_len = 0;
    for i in range(len(input_data_)):
        for j in range(len(input_data_[i])):
            get_words = input_data_[i][j].split(" ")
            if len(get_words) > max_len:
                max_len = len(get_word)
            for word in get_words:
                embedding_data = embedding_data + word
    dic = Word2vec(embedding_data, min_count = 1)
    for i in range(len(input_data_)):
        for j in range(len(input_data_[i])):
            get_words = input_data_[i][j].split(" ")
            embedding_list = []
            for word in get_words:
                vector = dic.wv[word]
                embed_len = len(vector)
                embedding_list.append(vector)
            if len(get_words) < max_len:
                for k in range(max_len - len(get_words)):
                    embedding_list.append(np.zero(embed_len))
            input_data[i][j] = embedding_list
    model = expansion(max_len, embed_len, len(output_data), 0.001)
    if os.path.exists(path + "/models/expansion.5h"):
        model.load_model(path + "/models/expansion.5h")
    else:
        model.train(input_data, output_data)
    testing_data =[] 
    for i in range(len(projects)):
        for bug in projects[i]:
            input_data_file = open(path + "/sbfl/ochiai/" + projects[i] + "/" + str(bug + 1) + "/" + "lines.txt")
            lines_testing = input_data_file.readlines()
            for line_test in lines_testing:
                line_test = line_test.split()
                file_test = line_test[0]
                test_line = line_test[1]
                test_data = open(file_test)
                test_data_lines = test_data.readlines()
                list_test = []
                for j in range(11):
                    line_test_ = test_data_lines[int(test_line)- 6 + i]
                    get_words = line_test.split()
                    embed_list_test = []
                    for word in get_words:
                        if word in dic.wv.keys():
                            vector = dic.wv[word]
                        else:
                            vector = np.zero(embed_len)
                        embed_list_test.append(vector)
                    list_test.append(embed_list_test)
                testing_data.append(list_test)
    output = model.predict(testing_data)
    # DataDepAnalysis from line 211 to line 221
    for i in range(len(input_data_)):
        for j in range(len(input_data_[i])):
            for k in range(len(input_data_[i][j])):
                if j == 6:
                    continue
                else:
                    regEx="[`~!@#$%^&*()_\\-+=|{}':;',\\[\\].<>/?~!@#?%��&*()��+|{}??�;:���?,??]"
                    if not re.match(regEx, input_data_[i][j][k]):
                        if input_data_[i][j][k] in input_data_[i][j]:
                            output[i][j] == 1
                            break
    output_file = open(path + "/data/expanded.txt","a")
    for i in range(len(testing_data)):
        for j in range(len(testing_data[i])):
            print(testing_data[i][j], file = output_file)
            print(output[i][j], file=output_file)
        print("=====", file = output_file)
    
    
            
        
def expansion(input_length, input_dim, output_dim, learning_rate):
    basic_input = Input(shape=(input_length, input_dim))
    rnn_output = GRU(units=hidden_dim, return_sequences=True)(basic_input)
    output = Dense(output_dim)(rnn_output)
    fixed_output = Softmax()(output)
    model = Model(basic_input, fixed_output)
    ada = adam(lr=learning_rate)
    model.compile(optimizer=ada, loss='categorical_crossentropy')
    return model

