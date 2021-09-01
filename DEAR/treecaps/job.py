import sys
import subprocess

directory = 'Result_B'
training = 0 # Use 1 for training and 0 for testing
n_classes = '10'
niter = '30'
lr = '0.001'
train_directory = "Dataset_B/train" 
test_directory = "Dataset_B/test"


if training==1:
    proc = subprocess.Popen(['python', 'main.py','--model',directory,'--n_classes',n_classes,'--niter',niter,'--train_directory',train_directory,'--test_directory',test_directory,'--training','--lr',lr])
    proc.wait()

#Uncomment to continue training with a lower learning rate
    lr = '0.00025'
    proc = subprocess.Popen(['python', 'main.py','--model',directory,'--n_classes',n_classes,'--niter',niter,'--train_directory',train_directory,'--test_directory',test_directory,'--training','--lr',lr])
    proc.wait()
    
else:
    proc = subprocess.Popen(['python', 'main.py','--model','Result_B_trained','--n_classes',n_classes,'--niter',niter,'--train_directory',train_directory,'--test_directory',test_directory,'--testing','--lr',lr])
    proc.wait()