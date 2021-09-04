import os
import time
import subprocess


rq = "all"
path = os.getcwd() + "/src"
start = time.process_time()
try:
    subprocess.call("python " + path + "/run_exp.py " + rq, shell=True)
except:
    print("Running Error!")
time_length = time.process_time() - start
print("Cost " + str(time_length) + " s to finish the model")
