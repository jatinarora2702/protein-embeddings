# Learning Dense Vector Representation for Proteins

This repo is largely developed on top of the [TAPE Repository](https://github.com/songlab-cal/tape). Please refer to it for setting up the basic python environment. For the DeepSF model, we provide the pretrained results over the test set in [deepsf](deepsf) directory.

## Overview

Protein sequences and their interactions can be seen as a natural language just like English. This opens up a wide spectrum of deep learning based language modeling techniques which can be applied to understand the inherent semantics of protein molecules. In this paper, we learn dense vector representations for proteins and compare them with popular sequence alignment methods on homology detection task. Results indicate that protein vector representations outperform alignment techniques by a significant margin in both supervised and unsupervised learning paradigms. Utilizing a fusion of pretrained protein vectors from TAPE (BERT) and DeepSF, we improve upon the existing state-of-the-art on SCOP 1.75 dataset by 5% in terms of accuracy. 

## Contributions

1. On the unsupervised clustering task, dense protein vectors are found to be significantly more effective than sequence alignment.
    
2. TSNE projection analysis reveals that protein embeddings learnt from TAPE (transformer-based architecture) are found to capture protein structure semantics much better than DeepSF (CNN-based architecture).
    
3. TAPE embeddings when combined with DeepSF embeddings are able to push the state-of-the-art on homology detection task over SCOP1.75 dataset by 5\%.

## Package Requirements

```
biopython
scikit-learn
pyclustering
tape
pytorch
transformers
matplotlib
scipy
numpy
```

## Dataset

Download [remote_homology](https://github.com/songlab-cal/tape#raw-data) json dataset from TAPE repository and unzip it in [data](data) directory.

## Global Sequence Alignment Based Clustering

```
python global_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10
python global_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10
```

## Vector Based Clustering

```
bash pretrained_cluster_script.sh
```

## Supervised Learning (Training)

For training from scratch:
```
CUDA_VISIBLE_DEVICES=0,2,4,5 python train.py transformer remote_homology --from_pretrained bert-base --batch_size 64 --gradient_accumulation_steps 16 --num_train_epochs 15
```

For resuming training from previously saved checkpoint:
```
CUDA_VISIBLE_DEVICES=0,2,4,5 python train.py transformer remote_homology --from_pretrained results/remote_homology_transformer_20-12-19-04-49-07_070319 --batch_size 64 --gradient_accumulation_steps 16 --num_train_epochs 15 --resume_from_checkpoint
```

## Supervised Learning (Evaluation)

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python test.py transformer remote_homology remote_homology_transformer_20-12-20-05-26-23_907518 --batch_size 64 --metric accuracy --split test_fold_holdout
```

## tSNE Projections

```
python tape_cluster.py --name remote_homology --suffix test_fold_holdout --cluster_label class_label
python deepsf_cluster.py --cluster_label class_label
```

## Result Analysis

```
python result_analysis.py
```

## Acknowledgements
This work is developed as a course project for [CS466: Introduction to Bioinformatics course](http://www.el-kebir.net/teaching/CS466.html). I whole-heartedly thank our course instructor, Prof. Mohammed El-Kebir and all TAs and course staff for their teaching, guidance and continuous support. The work is summarized with detailed results and analysis in the [submitted report](CS466_ProjectReport.pdf).

## Contact Details

```
Jatin Arora
jatin2@illinois.edu
```
