# Learning Dense Vector Representation for Proteins

This repo is largely developed on top of the [TAPE Repository](https://github.com/songlab-cal/tape). Please refer to it for setting up the basic python environment. For the DeepSF model, we provide the pretrained results over the test set in [deepsf](deepsf) directory.

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

## Contact Details

```
Jatin Arora
jatin2@illinois.edu
```
