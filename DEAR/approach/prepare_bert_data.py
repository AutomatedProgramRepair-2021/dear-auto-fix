import os
import sys
import shutil
import argparse
import tempfile
from urllib.request import urlretrieve
import zipfile
import io


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass


def format_data(data_dir, path_to_data):
    print("Processing data...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "training.txt")
        mrpc_test_file = os.path.join(path_to_data, "testing.txt")
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file

    with io.open(mrpc_test_file, encoding='utf-8') as data_fh, \
            io.open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding='utf-8') as test_fh:
        body = data_fh.readlines()
       # test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for i in range(len(body)):
            try:
                if i ==0:
                    continue
                body[i] = body[i].strip().split('\t')
                while '' in body[i]:
                    body[i].remove('')
                for j in range(len(body[i])):
                    body[i][j] = body[i][j].strip()
                body[i] = "\t".join(body[i])
                label, id1, id2, s1, s2 = body[i].strip().split('\t')
                if is_number(label) and is_number(id1) and is_number(id2):
                    test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (i, id1, id2, s1, s2))
            except:
                print(i)
                print(body[i].strip().split('\t'))



    with io.open(mrpc_train_file, encoding='utf-8') as data_fh, \
         io.open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding='utf-8') as train_fh, io.open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding='utf-8') as dev_fh:
        body = data_fh.readlines()
       # train_fh.write(header)
        for i in range(len(body)):
            try:
                if i == 0:
                    continue
                body[i] = body[i].strip().split('\t')
                while '' in body[i]:
                    body[i].remove('')
                for j in range(len(body[i])):
                    body[i][j] = body[i][j].strip()
                body[i] = "\t".join(body[i])
                label, id1, id2, s1, s2 = body[i].strip().split('\t')
                if is_number(label) and is_number(id1) and is_number(id2):
                    if i <=len(body):
                        train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
                    else:
                        dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            except:
                print(body[i].strip().split('\t'))

                
    print("\tCompleted!")
    
    
path_ = os.path.abspath(os.path.dirname(os.getcwd()))
format_data(path_ + "/bert_data", path_ + "/bert_data")