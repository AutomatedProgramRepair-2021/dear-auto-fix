# DEAR: A Novel Deep Learning-based Approach for Automated Program Repair

<p aligh="center"> This repository contains the code for <b>DEAR: A Novel Deep Learning-based Approach for Automated Program Repair</b> and the Page (https://automatedprogramrepair-2021.github.io/DEAR-Public/) that has some visualizd data. </p>

# Experimental Results and Data: https://automatedprogramrepair-2021.github.io/DEAR-Public/
# Source Code: https://github.com/AutomatedProgramRepair-2021/dear-auto-fix#Instruction_to_Run_DEAR
# Demo: https://github.com/AutomatedProgramRepair-2021/dear-auto-fix#Demo

## Contents
## 1. [Dataset](#Dataset)
## 2. [Requirement](#Requirement)
## 3. [Editable Parameters](#Editable-Parameters)
## 4. [Instruction_to_Run_DEAR](#Instruction_to_Run_DEAR)
## 5. [Demo](#Demo)
## 6. [Reference](#Reference)


## Introduction

We present DEAR, a DL-based approach that supports auto-fixing for the bugs that require changes at once to one or multiple hunks and one or multiple consecutive statements.We first design a novel fault localization (FL) technique for multi-hunk, multi-statement fixes that combines traditional  spectrum-based (SB) FL with deep learning and data-flow analysis. It takes the buggy statements returned by the SBFL, and detects the buggy hunks to be fixed at once and expands a buggy statement s in a hunk to include other suspicious statements from s. We enhance a two-tier, tree-based LSTM model that incorporates cycle training and uses a divide-andconquer strategy to learn proper code transformations for fixing multiple statements in the suitable fixing context consisting of surrounding subtrees. We conducted several experiments to evaluate DEAR on three datasets: Defects4J (395 bugs), BigFix (+26k bugs), and CPatMiner (+44k bugs). In CPatMiner, DEAR fixes 71 and 164 more bugs, including 52 and 61 more multi-hunk/multistatement bugs, than existing DL-based APR tools. Among 667 fixed bugs, there are 169 (25.3%) multi-hunk/multi-statement ones. On Defects4J, it outperforms those tools from 42–683% in terms of the number of auto-fixed bugs with only Top-1 ranked patches.

## Dataset

The Dataset we used in the paper:

Defects4J[1]: https://github.com/rjust/defects4j

BigFix[2]: https://drive.google.com/open?id=1KL3M-BbisVLWXyvn05V6huSLNUby_9qN

CPatMiner[3]: https://drive.google.com/open?id=1M_0dRYqhCMh26GQbnX4Igp_2jSrTS1tV 

Before using our tool, please download the two clearned version of data and also setup the Defects4J. You can use /DEAR/approach/sbfl/setup.sh to install Defects4J or manually install it, then you need to setup Defects4J and setup enviroment variable ```D4J_HOME``` and ```DEFECTS4J_HOME``` that both point to the install path of Defects4J.

## Requirement

#### Javac = 1.8
#### Java = 1.7
#### Ant = 1.9
#### Python <= 3.6
#### tensorflow >= 1.11.0

Please check all other required packages in the [requirement.txt](https://github.com/AutomatedProgramRepair-2021/dear-auto-fix/tree/main/DEAR/approach/requirements.txt)

## Editable-Parameters

1. System variables ```defects4J``` and ```D4J_HOME``` are needed to predefine. The ```defects4J``` is point to the ```path2defects4j/framework/bin```, and ```D4J_HOME``` is point to ```path2defects4j```

2. In the source code, we use os.environ["CUDA_VISIBLE_DEVICES"] to control the GPU we want to use to run the deep learning model. So if you have multiple GPUs to use and want to control which GPU you want to use, please change the GPU number in this command in the source code.

3. The parameter ```BERT_BASE_DIR``` in the ```run_bert.sh``` need to be changed to the address that you put the pre-trained bert model.

4. The parameter ```rq``` in the ```main.py``` can be modified for different experiments.

## Instruction_to_Run_DEAR
1. Download BigFix and CPatMiner from the given link in [Dataset](#Dataset) to the directory DEAR/data (You need to create the ```data``` folder). Feel free to reduce the size for a small demo.

2. If you have setup ```D4J_HOME```, check if ```DEFECTS4J_HOME``` is also set to point to ```path2defects4j```

3. Run ```DEAR/approach/sbfl/setup.sh```

4. Run ```get_fl_data.py``` in directory  ```DEAR/approach```

5. Run ```prepare_bert_data.py```

6. Download the [Pre-train Model](https://github.com/google-research/bert#pre-trained-models)  from Bert [4] and change ```BERT_BASE_DIR``` in  ```DEAR/bert/run_bert.sh``` to point to the model just downloaded.

7. Run ```./run_bert.sh``` in directory ```Dear/bert```

8. Run ```main.py```in directory  ```DEAR/approach```

## Demo

The requirements packages for the demo can also be found in the [requirement.txt](https://github.com/AutomatedProgramRepair-2021/dear-auto-fix/tree/main/DEAR/approach/requirements.txt)

Tested with {Keras==2.6.0, tensorflow==2.6.0} and {Keras==2.4.3, tensorflow==1.14}. If you are facing problem, please check your package version.

For the testing purpose of running, please download our demo that contains the model for fixing a small set of bugs. Demo download: https://drive.google.com/file/d/1pgIX6OlCNVhGBj5ybpyXj6vhFOgLVRZN/view?usp=sharing

Put ```model``` in ```DEAR/approach```, ```demo``` in ```DEAR/data``` (You need to create the ```data``` folder)

change model path in ```DEAR/approach/src/model.py``` at line 170 if you are using Linux

run ```run_demo.py``` to see the results.

This demo only contains the fixing part of the project due to the long period of the runtime for full project. The demo is running on a big dataset and follows the same setting in the RQ1 in our paper.

## Reference

[1] René Just, Darioush Jalali, and Michael D Ernst. 2014. Defects4J: A database of existing faults to enable controlled testing studies for Java programs. In Proceedings of the 2014 International Symposium on Software Testing and Analysis. 437–440.

[2] Yi Li, ShaohuaWang, Tien N. Nguyen, and Son Van Nguyen. 2019. Improving Bug Detection via Context-Based Code Representation Learning and Attention-Based Neural Networks. Proc. ACM Program. Lang. 3, OOPSLA, Article Article 162 (Oct. 2019), 30 pages. https://doi.org/10.1145/3360588

[3] Hoan Anh Nguyen, Tien N. Nguyen, Danny Dig, Son Nguyen, Hieu Tran, and Michael Hilton. 2019. Graph-Based Mining of in-the-Wild, Fine-Grained, Semantic Code Change Patterns. In Proceedings of the 41st International Conference on Software Engineering (ICSE ’19). IEEE Press, 819–830. https://doi.org/10.1109/ICSE.2019.00089

[4] Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805. https://arxiv.org/abs/1810.04805
