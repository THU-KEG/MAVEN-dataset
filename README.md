# MAVEN-dataset
Source code and dataset for EMNLP 2020 paper "MAVEN: A Massive General Domain Event Detection Dataset".

## Data

The dataset (ver. 1.0) can be obtained from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/874e0ad810f34272a03b/) or [Google Drive](https://drive.google.com/drive/folders/19Q0lqJE6A98OLnRqQVhbX3e6rG4BVGn8?usp=sharing). The data format is introduced in [this document](DataFormat.md).

We also release the document topics for data analysis and model development. The [``docid2topic.json``](docid2topic.json) is to map the document ids to their EventWiki topic labels.

## CodaLab

To get the test results, you can submit your predictions to our permanent [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/395) (the [older version](https://competitions.codalab.org/competitions/27320) will be phased out soon). For the evaluation method, please refer to the [evaluation script](evaluate.py).

## Codes

We release the source codes for the baselines, including [DMCNN](baselines/DMCNN_BiLSTM_(CRF)), [BiLSTM](baselines/DMCNN_BiLSTM_(CRF)), [BiLSTM+CRF](baselines/DMCNN_BiLSTM_(CRF)), [MOGANED](baselines/MOGANED) and [DMBERT](baselines/DMBERT).

## Citation

If these data and codes help you, please cite this paper.

```bib
@inproceedings{wang2020MAVEN,
  title={{MAVEN}: A Massive General Domain Event Detection Dataset},
  author={Wang, Xiaozhi and Wang, Ziqi and Han, Xu and Jiang, Wangyi and Han, Rong and Liu, Zhiyuan and Li, Juanzi and Li, Peng and Lin, Yankai and Zhou, Jie},
  booktitle={Proceedings of EMNLP 2020},
  year={2020}
}
```



