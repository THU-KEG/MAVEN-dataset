# DMBERT
This code is the implementation for [DMBERT](https://www.aclweb.org/anthology/N19-1105/) model. The implementations are based on [Huggingface's Transformers](https://github.com/huggingface/transformers), especially its example for the multiple-choice task.



## Requirements

- python==3.6.9

- torch==1.2.0

- transformers==2.8.0

- sklearn==0.20.2

  

## Usage

Hint: please read and delete all the comments after ```\``` in each line of the ```.sh``` scripts before running them.

### On MAVEN:

1. Download MAVEN data files.
2. Run ```run_MAVEN.sh``` for training and evaluation on the devlopment set.  
3. Run ```run_MAVEN_infer.sh``` to get predictions on the test set (dumped to ```results.jsonl```).

See the two scripts for more details.

### On ACE

1. Preprocess ACE 2005 dataset as in [this repo](https://github.com/thunlp/HMEAE).
2. Run ``run_ACE.sh`` for training and evaluation.
