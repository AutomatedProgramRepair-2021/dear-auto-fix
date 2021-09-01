import os
import time
import subprocess


rq = "1"
path = os.getcwd() + "/src"
start = time.process_time()
try:
    subprocess.call("python3 " + path + "/run_exp.py " + rq, shell=True)
except:
    print("Running Error!")
time_length = time.process_time() - start
print("Cost " + str(time_length) + " s to finish the model")
