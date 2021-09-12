import numpy as np
import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity


def set_threshold(input_list, threshold):
    fixed_list = []
    for i in range(len(input_list)):
        if input_list[i] > threshold:
            fixed_list.append(1)
        else:
            fixed_list.append(0)
    return fixed_list


def get_score(input_a, input_b):
    cos = cosine_similarity(input_a, input_b)
    if input_a.all() == input_b.all():
        cos = 1
    cos = np.array(cos)
    cos = np.mean(cos)
    if cos < 0:
        cos = cos * -1
    if cos < 0.05:
        cos = cos + 0.05
    return cos


def evaluation(predicted_d, dataset):
    if dataset != "cpatminer-defects4j":
        evaluation_data = "../data/" + dataset + "/testing_lable.npy"
        evaluation_d = np.load(evaluation_data)
        y_true = evaluation_d.flatten()
        y_pred = predicted_d.flatten()
        y_pred_fix = set_threshold(y_pred, y_pred[0])
        accuracy = sk.metrics.accuracy_score(y_true, y_pred_fix)
        print("Accuracy: " +  accuracy)
    else:
        projects = ["Chart", "Closure", "Lang", "Math", "Mockito", "Time"]  
        bugs = [26, 133, 65, 106, 38, 27]
        path= os.getcwd()
        total = 1
        for i in range(len(projects)):
            for bug in bugs[i]:
                input_file = open(path + "/data/expanded" + project[i] + int(bug+1) + ".txt","a")
                file_ = input_file.readline.split(",")[0]
                line_num = input_file.readline.split(",")[1]
                change_file = open(file_, "w")
                input_lines = input_file.readlines()
                input_lines[line_num] = predicted_d[total]
                print(input_lines, file = change_file) #apply patches by replacing the buggy statements with the fixed ones
                folder = path + "/sbfl/defects4j_data/" + projects[i] + "/" + str(bug + 1)
                subprocess.call("./sbfl.sh " + projects[i] + " " + str(bug + 1) + " " + folder, shell=True) #compiling patched programs and calling tests
                total = total + 1