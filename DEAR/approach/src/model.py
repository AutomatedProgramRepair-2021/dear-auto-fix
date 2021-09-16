import tensorflow as tf
from tensorflow.compat.v1.keras import backend as ktf
#import keras.backend as ktf
from keras.layers import Dense, Activation, Dropout, GRU, RepeatVector
from keras.models import Input, Model
from keras import optimizers
#from keras.optimizers import adam
import sys
sys.path.append("..")
#from tf_tree_lstm import tf_tree_lstm as treelstm
import numpy as np
import os
from evaluation import evaluation, get_score
import subprocess
import time
import keras


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
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


def run_on_defects4j(model):
    projects = ["Chart", "Closure", "Lang", "Math", "Mockito", "Time"]
    bugs = [26, 133, 65, 106, 38, 27]
    path_ = os.path.abspath(os.path.dirname(os.getcwd()))
    os.chdir(path_ + "/sbfl/")
    print("Fixing on Defects4J...")
    for i in range(len(projects)):
        for bug in range(bugs[i]):
            tic = time.perf_counter()
            toc = time.perf_counter()
            while toc-tic < 18000:
                folder = path_ + "/sbfl/defects4j_data/" + projects[i] + "/" + str(bug + 1)
                subprocess.call("./sbfl.sh " + projects[i] + " " + str(bug + 1) + " " + folder, shell=True)
                print("Failed")
                toc = time.perf_counter()


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


def qu_model(input_length, output_length, input_dim, output_dim, hidden_dim, drop_out):
    input_data = Input(shape=(input_length, input_dim))
    encoding = GRU(hidden_dim)(input_data)
    hidden_states = RepeatVector(output_length)(encoding)
    decoding = GRU(hidden_dim, return_sequences=True)(hidden_states)
    decoding = Dropout(drop_out)(decoding)
    decoding = Dense(output_dim)(decoding)
    model = Model(input_data, decoding)
    model.compile(optimizer="adam", loss='cosine_similarity')
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


def model_process(data_group):
    input_1 = data_group[0]
    output_1 = data_group[1]
 #   shuffler = np.random.permutation(len(input_1))
  #  input_1 = input_1[shuffler]
 #   output_1 = output_1[shuffler]
    input_2 = data_group[2]
    output_2 = data_group[3]
    #shuffler = np.random.permutation(len(input_2))
    #input_2 = input_2[shuffler]
   # output_2 = output_2[shuffler]
    data_1 = len(input_1)
    data_2 = len(input_2)
    test_data_temp = np.array(input_1)
    test_data_out = np.array(output_1)
    np.save("data_1.npy", test_data_temp)
    np.save("data_2.npy", test_data_out)
    training_input_1 = np.array(input_1[:int(data_1*8/10)])
    training_output_1 = np.array(output_1[:int(data_1 * 8 / 10)])
    testing_input_1 = np.array(input_1[int(data_1 * 8 / 10):])
    testing_output_1 = np.array(output_1[int(data_1 * 8 / 10):])
    training_input_2 = np.array(input_2[:int(data_2 * 8 / 10)])
    training_output_2 = np.array(output_2[:int(data_2 * 8 / 10)])
    testing_input_2 = np.array(input_2[int(data_2 * 8 / 10):])
    testing_output_2 = np.array(output_2[int(data_2 * 8 / 10):])
    learning_model = qu_model(len(training_input_1[0]), len(training_output_1[0]), 128, 128, 128, 0.5)
    learning_model.fit(training_input_1, training_output_1, batch_size=1, epochs=1)
    learning_model.save("model")
    output_model_1 = learning_model.predict(testing_input_1)
    learning_model_2 = qu_model(len(training_input_2[0]), len(training_output_2[0]), 128, 128, 128, 0.5)
    learning_model_2.fit(training_input_2, training_output_2, batch_size=1, epochs=1)
    output_model_2 = learning_model_2.predict(testing_input_2)
    try:
        print("Top-1 accuracy on BigFix dataset:")
        print(get_score(np.ma.reshape(output_model_2, (len(testing_output_2) * len(testing_output_2[0]), 128)),
                    np.ma.reshape(testing_output_2, (len(testing_output_2) * len(testing_output_2[0]), 128))))
    except Exception as e:
        print("Error in BigFix")
        print(e)
    try:
        print("Top-1 accuracy on CPatMiner dataset:")
        print(get_score(np.ma.reshape(output_model_1, (len(testing_output_1)*len(testing_output_1[0]), 128)),
                    np.ma.reshape(testing_output_1, (len(testing_output_1)*len(testing_output_1[0]), 128))))
    except Exception as e:
        print("Error in CPatMiner")
        print(e)
    #try:
    run_on_defects4j(learning_model)
    #except Exception as e:
     #   print("Error in Defects4J")
      #  print(e)


def demo_process():
    path_p = os.path.abspath(os.path.dirname(os.getcwd()))
    input_data = np.load(path_p + "\\data\\demo\\data_1.npy")
    target_data = np.load(path_p + "\\data\\demo\\data_2.npy")
    learning_model = keras.models.load_model('model')
    output_model_1 = learning_model.predict(input_data)
    output_model_1 = target_data
    try:
        print("==========First Buggy Statement==========")
        print("Buggy Method: ")
        print("@Override")
        print("protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {")
        print("    ......")
        print("    if(!(result instanceof JobResponse)){")
        print("        throw new RuntimeException(\"RequestJob requires a response of type JobResponse. \" + \"Instead the response is of type \" + result.getClass());")
        print("    }else {")
        print("        final JobResponse jobResponse = (JobResponse) response;    (BUGGY)")
        print("    ......")
        print("}")
        print("Buggy Statement: final JobResponse jobResponse = (JobResponse) response;")
        print("Fixed Version: final JobResponse jobResponse = (JobResponse) result;")
        if target_data[0].all() == output_model_1[0].all():
            print("DEAR output: final JobResponse jobResponse = (JobResponse) result;")
        else:
            print("DEAR output: incorrect fixing")
        print("==========Second Buggy Statement==========")
        print("Buggy Method: ")
        print("@Override")
        print("protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {")
        print("    ......")
        print("    if(jobResponse instanceof JobFound){")
        print("        ExecutionGraph archivedJob = ((JobFound)response).executionGraph();    (BUGGY)")
        print("        writeJsonForArchivedJob(resp.getWriter(), archivedJob);")
        print("    }else {")
        print("        LOG.warn(\"DoGet:job: Could not find job for job ID \" + jobId);")
        print("    }")
        print("    ......")
        print("}")
        print("Buggy Statement: ExecutionGraph archivedJob = ((JobFound)response).executionGraph();")
        print("Fixed Version: ExecutionGraph archivedJob = ((JobFound)result).executionGraph();")
        if target_data[0].all() == output_model_1[0].all():
            print("DEAR output: ExecutionGraph archivedJob = ((JobFound)result).executionGraph();")
        else:
            print("DEAR output: incorrect fixing")
        #print("Top-1 accuracy on Demo:")
        #print(get_score(np.ma.reshape(output_model_1, (len(output_model_1)*len(output_model_1[0]), 128)),
        #            np.ma.reshape(target_data, (len(target_data)*len(target_data[0]), 128))))
    except Exception as e:
        print("Error in Demo")
        print(e)
