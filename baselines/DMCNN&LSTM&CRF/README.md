# DMCNN & LSTM & LSTM+CRF
The code is an **unofficial** implementation of [Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks](https://www.aclweb.org/anthology/P15-1017/) as well as LSTM, LSTM+ CRF for event detection and classification. 
 
# Requirements
+ pytorch-gpu >= 1.6 & CUDA >= 10.2
+ numpy
+ sklearn
+ seqeval
+ tqdm

# Usage
To run this code, you need to:
1. put files of MAVEN dataset in `./raw`
2. run ```python main.py --config [path of config file] --gpu [gpu, optional]```  
we will train, evaluate and test on separate files in every epoch. We output the performance of training and evaluating, while generating result file of testing for [Codelab](https://competitions.codalab.org/competitions/27320#learn_the_details-submission-format).

All parameters are in config files in `./config/`, you can modify them as you wish.

# Note
1. We use BIO schema in LSTM+CRF, and our experiment results published in [MAVEN: A Massive General Domain Event Detection Dataset](https://arxiv.org/abs/2004.13590) was tested based on it. Now that the evaluating script in [MAVEN-dataset Repo](https://github.com/THU-KEG/MAVEN-dataset/blob/main/evaluate.py) hasn't support BIO test, our result file of LSTM+CRF cannot support [Codelab](https://competitions.codalab.org/competitions/27320#learn_the_details-submission-format) and our `type_id` in it is not official. Detail information can be seen in files generated in `./data/`.