import os
import sys
import shutil
import argparse
import tempfile
from urllib.request import urlretrieve
import zipfile
import io


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
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            try:
                label, id1, id2, s1, s2 = row.strip().split('\t')
            except:
                print(idx)
                print(row.strip().split('\t'))
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))


    with io.open(mrpc_train_file, encoding='utf-8') as data_fh, \
         io.open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding='utf-8') as train_fh:
        header = data_fh.readline()
        train_fh.write(header)
        for row in data_fh:
            try:
                label, id1, id2, s1, s2 = row.strip().split('\t')
            except:
                print(row.strip().split('\t'))
            train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
                
    print("\tCompleted!")
    
    
path_ = os.path.abspath(os.path.dirname(os.getcwd()))
format_data(path_ + "/bert_data", path_ + "/bert_data")