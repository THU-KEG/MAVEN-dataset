# DMBERT
This code is the implementation for BERT+CRF model. The implementations are based on [Huggingface's Transformers](https://github.com/huggingface/transformers) and the BERT+CRF implementations in [this repo](https://github.com/mezig351/transformers/tree/ner_crf/examples/ner).



## Requirements

- python==3.6.9

- torch==1.2.0

- transformers==2.6.0

- sklearn==0.20.2

- seqeval

  

## Usage

Hint: please read and delete all the comments after ```\``` in each line of the ```.sh``` scripts before running them.

### On MAVEN:
The codes are in the ```BERT-CRF-MAVEN``` folder.

1. Download MAVEN data files.
2. Run ```run_MAVEN.sh``` for training and evaluation on the devlopment set.  
3. Run ```run_MAVEN_infer.sh``` to get predictions on the test set (dumped to ```OUTPUT_PATH/results.jsonl```).

See the two scripts for more details.

### On ACE
The codes are in the ```BERT-CRF-ACE``` folder.

1. Preprocess ACE 2005 dataset as in [this repo](https://github.com/thunlp/HMEAE).
2. Run ``run_ACE.sh`` for training and evaluation.
