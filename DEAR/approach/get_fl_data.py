import os
import subprocess
from operator import itemgetter
import random

path = os.getcwd()
path_p = os.path.abspath(os.path.dirname(os.getcwd()))
projects = ["Chart", "Closure", "Lang", "Math", "Mockito", "Time"]
bugs = [26, 133, 65, 106, 38, 27]
folder = path + "/sbfl/defects4j_data"
subprocess.call("mkdir " + folder, shell=True)
for i in range(len(projects)):
    folder = path + "/sbfl/defects4j_data/" + projects[i]
    subprocess.call("mkdir " + folder, shell=True)
if not os.path.exists(path_p + "/bert_data/"):
    os.makedirs(path_p + "/bert_data/")
output_bert = open(path_p + "/bert_data/testing.txt", "a")
print("Quality \t #1 ID \t #2 ID \t #1 String \t #2 String", file=output_bert)
ID_num = 0
for i in range(len(projects)):
    for bug in range(bugs[i]): 
        try:
            folder = path + "/sbfl/defects4j_data/" + projects[i] + "/" + str(bug + 1)
            subprocess.call("mkdir " + folder, shell=True)
            subprocess.call("defects4j checkout -p " + projects[i] + " -v " + str(bug + 1) + "b -w " + folder, shell=True)
            os.chdir(path + "/sbfl/")
            subprocess.call("./sbfl.sh " + projects[i] + " " + str(bug + 1) + " " + folder, shell=True)
            input_file = open(path + "/sbfl/ochiai/" + projects[i] + "/" + str(bug + 1) + "/stmt-susps.txt")
            lines = input_file.readlines()
            scores = [] 
            for m in range(len(lines)):
                if m != 0:
                    score = lines[m].split(",")[1].strip()
                    scores.append(float(score))
            output = [index for index, value in sorted(enumerate(scores), key=itemgetter(1))]
            Code = []
            ID = []
            for j in range(100):
                pick_out = int(output[len(output) - 1 - j]) + 1
                pick_data = lines[pick_out]
                file_path_ = pick_data.split(",")[0].strip()
                file_name = file_path_.split("#")[0].strip()
                file_name = file_name.replace(".", "/")
                file_name = file_name + ".java"
                line_num = file_path_.split("#")[1].strip()
                for path_, dirs, files in os.walk(path + '/sbfl/defects4j_data/' + projects[i] + '/' + str(bug + 1)):
                    if files:
                        for name in files:
                            if name.endswith(".java"):
                                file_path = os.path.join(path_, name)
                                if file_name in file_path:
                                    source_code = open(file_path)
                                    code_lines = source_code.readlines()
                                    code_line = code_lines[int(line_num) - 1].strip()
                                    output_ = file_path + "," + line_num + "," + code_line
                                    Code.append(code_line)
                                    ID_num = ID_num + 1
                                    ID.append(ID_num)
                                    output_file_1 = open(
                                        path + "/sbfl/ochiai/" + projects[i] + "/" + str(bug + 1) + "/" + "lines.txt", "a")
                                    print(output_, file = output_file_1)
                                    break

            for l in range(len(Code)):
                for k in range(len(Code) - l - 1):
                    print(0, "\t", ID[l], "\t", ID[k + l + 1], "\t", Code[l], " .\t", Code[k + l + 1] + " .", file=output_bert)
        except:
            error_file = open("error_report.txt","a")
            print(projects[i] + str(bug + 1), file = error_file)
            continue  

train_bert = open(path_p + "/bert_data/training.txt", "a")
print("Quality \t #1 ID \t #2 ID \t #1 String \t #2 String", file=train_bert)
ID_num = 0 
for i in range(44154):
    try:
        bug_id = i + 1
        source_file_root = path_p + "/data/CPatMiner_/" + str(bug_id) + "/"
        label_file = open(source_file_root + "removed_lines.txt")
        label_lines = label_file.readlines()
        mark = 0
        var_exists = 'var' in locals() or 'var' in globals()
        if var_exists:
            if len(Code):
                mark = 1
                old_code = Code
                old_id = ID
        Code = []
        ID = []
        for j in range(len(label_lines)):
            if not label_lines[j].strip().isnumeric():
                file_name_ = label_lines[j].strip()
            else:
                line_num = label_lines[j].strip()
                new_file_read = open(source_file_root + "before/" + file_name_)
                lines_data = new_file_read.readlines()
                Code.append(lines_data[int(line_num)-1].strip())
                ID_num = ID_num + 1
                ID.append(ID_num)
        total = 0
        for l in range(len(Code)):
            for k in range(len(Code) - l - 1):
                print(1, "\t", ID[l], "\t", ID[k + l + 1], "\t", Code[l], " .\t", Code[k + l + 1] + " .", file=train_bert)
                total = total + 1
        if mark:
            for k in range(total):
                a = random.randint(0,len(old_copy)-1)
                b = random.randint(0,len(Copy)-1)
                print(0, "\t", old_id[a], "\t", ID[b], "\t", old_code[a], " .\t", Code[b] + " .", file=train_bert)              
    except:
        continue
            
        