import sys
import process
import model
import subprocess
import expansion

def main(argv):
    expansion.run_expansion()
    if len(argv > 1):
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
            print("Incorrect RQ number!")
            exit()
    if rq == "1":
        model.run_model("cpatminer-defects4j")
    if rq == "2":
        model.run_model("cpatminer")
        model.run_model("bigfix")
    if rq == "3":
        model.run_model("cpatminer-defects4j")
    

if __name__ == "__main__":
    main(sys.argv[1:])