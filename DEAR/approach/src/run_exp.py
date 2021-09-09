import sys
import process
import model
import subprocess
import numpy as np
#from expansion import expansion

def main(argv):
    #expansion.run_expansion()
    rq = "all"
    if len(argv) > 1:
        print("Only one paramter please")
        exit()
    else:
        for item in argv:
            rq = item
    if rq == "1" or rq == "3":
        process.preprocess("cpatminer")
    else:
        if rq == "2":
            process.preprocess("cpatminer")
            process.preprocess("bigfix")
        else:
            if rq != "all" and rq != "demo":
                print("Incorrect RQ number!")
                exit()
    if rq == "1":
        model.run_model("cpatminer-defects4j")
    if rq == "2":
        model.run_model("cpatminer")
        model.run_model("bigfix")
    if rq == "3":
        model.run_model("cpatminer-defects4j")
    if rq == "all":
        output = process.process_data()
        cleaned = process.clean_data(output[0], output[1], output[2], output[3])
        prepareed_data = process.data_prepare(cleaned)
        model.model_process(prepareed_data)
    if rq == "demo":
        model.demo_process()
    

if __name__ == "__main__":
    main(sys.argv[1:])