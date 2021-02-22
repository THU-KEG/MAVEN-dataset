# DMCNN & BiLSTM & BiLSTM+CRF
The codes are implementations of [DMCNN](https://www.aclweb.org/anthology/P15-1017/), BiLSTM and BiLSTM+CRF for event detection on MAVEN. 

## Requirements

+ torch==1.6
+ CUDA==10.2
+ numpy
+ sklearn
+ seqeval==1.2.2
+ tqdm==4.44.0

## Usage

To run this code, you need to:
1. put raw files of MAVEN dataset in `./raw`
2. run ```python main.py --config [path of config files] --gpu [gpu, optional]```  
we will train, evaluate and test models in every epoch. We output the performance of training and evaluating, and generate test result files for submit to [CodaLab](https://competitions.codalab.org/competitions/27320#learn_the_details-submission-format).

All the hyper-parameters for the three models are in config files at `./config/`, you can modify them as you wish.
