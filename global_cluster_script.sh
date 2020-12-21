#!/bin/bash

# DeepSF-Coarse
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach euclid-pca
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach euclid-tsne
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach cosine-pca
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach cosine-tsne
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach kldiv-pca
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach kldiv-tsne

# # DeepSF-Fine
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach euclid-pca
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach euclid-tsne
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach cosine-pca
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach cosine-tsne
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach kldiv-pca
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name deepsf_remote_homology --approach kldiv-tsne

# TAPE-Coarse
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach euclid-pca
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach euclid-tsne
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach cosine-pca
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach cosine-tsne
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach kldiv-pca
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach kldiv-tsne

# TAPE-Fine
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach euclid-pca
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach euclid-tsne
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach cosine-pca
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach cosine-tsne
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach kldiv-pca
python pretrained_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10 --name remote_homology --approach kldiv-tsne
