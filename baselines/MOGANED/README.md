# MOGANED-Implementation
The code is an **unofficial** implementation of [Event Detection with Multi-order Graph Convolution and Aggregated Attention](https://www.aclweb.org/anthology/D19-1582/) (EMNLP 2019 paper). 

# Requirments
tensorflow-gpu==1.10 & CUDA 9 (or tensorflow-gpu==1.14 & CUDA 10.0)

stanfordcorenlp (see https://github.com/Lynten/stanford-corenlp for detail)

numpy

tqdm

# Usage
To run this code, you need to:
1. modify MAVEN dataset path, GloVe file path and stanfordcorenlp path in ```constant.py```
2. Run ```python train.py --gpu [YOUR_GPU] --mode MOGANED --eval False``` to train.  Run ```python train.py --gpu [YOUR_GPU] --mode MOGANED --eval True``` to get prediction on test set (dumped to ```results.jsonl```).

All parameters are in ```constant.py```, you can modify them as you wish.

# About Preprocess
When you first run this code, the code will do preprocessing. The preprocessing is quite low and may take a whole night (so run it and you can go to sleep!). This is because getting dependency trees are quite slow.

However, preprocessing will only run once and the preprocessed files will be dumped to the maven dataset path. Next time you run the code the code will read them and won't do any more preprocessing.

# Results on MAVEN
We run this code and submit the results to the leaderboard (username: wzq016):
|Method|Precision|Recall|F1|
|--|--|--|--|
|MOGANED(Paper)|63.4|64.1|63.8|
|MOGANED(Leaderboard)|64.6|66.0|65.3|

P.S. We improved the implementation after paper was published so the score is higher now.

# Note
There are some differences on training strategy between this code and the original paper:
1. The code doesn't use BIO schema. This is because trigger words are usually a single word rather than a phrase in ACE05, this won't affect results in ACE05.
2. The code doesn't use L2-norm, only use dropout. 
3. The code uses AdamOptimizer rather than AdadeltaOptimizer. During experiments, I found Adadelta can't train a good classifier, however, Adam can. 
4. This code sets bias loss lambda to 1 rather than 5 since I found this will make F1 score higher.

# Run it on ACE 2005
Please see [this repo](https://github.com/wzq016/MOGANED-Implementation).

