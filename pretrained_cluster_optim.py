import argparse
import json
import pickle

import numpy as np
import math
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial import distance
from scipy.special import rel_entr, softmax

from sklearn.metrics import adjusted_rand_score
from numpy.random import default_rng

rng = default_rng()


def main(args):
    with open("saved/{0}_{1}_{2}".format(args.name, args.suffix, args.cluster_label), "rb") as handle:
        plot_data = pickle.load(handle)

    K = len(plot_data["label"])
    
    # print("full data len: ", K)
    print("dataset name: ", args.name)
    print("approach: ", args.approach)
    
    tsne_data = np.hstack([np.array(plot_data["x"]).reshape(-1, 1), np.array(plot_data["y"]).reshape(-1, 1)])

    softmax_pca_data = softmax(plot_data["pca"], axis=1)
    softmax_tsne_data = softmax(tsne_data, axis=1)

    dp = [[0.0 for j in range(K)] for i in range(K)]
    for i in range(K):
        for j in range(i+1, K):

            if args.approach == "euclid-pca":
                dist = distance.euclidean(plot_data["pca"][i], plot_data["pca"][j])
            elif args.approach == "euclid-tsne":
                dist = distance.euclidean(tsne_data[i], tsne_data[j])
            elif args.approach == "cosine-pca":
                dist = distance.cosine(plot_data["pca"][i], plot_data["pca"][j])
            elif args.approach == "cosine-tsne":
                dist = distance.cosine(tsne_data[i], tsne_data[j])
            elif args.approach == "kldiv-pca":
                dist = sum(rel_entr(softmax_pca_data[i], softmax_pca_data[j]) + rel_entr(softmax_pca_data[j], softmax_pca_data[i]))
            elif args.approach == "kldiv-tsne":
                dist = sum(rel_entr(softmax_tsne_data[i], softmax_tsne_data[j]) + rel_entr(softmax_tsne_data[j], softmax_tsne_data[i]))

            dp[i][j] = dp[j][i] = dist
    print("created dist matrix")

    labels = plot_data["label"][:K]
    
    cluster_count = len(set(labels))
    print("num clusters: ", cluster_count)
    inits = rng.choice(K, size=cluster_count, replace=False)
    # print("cluster inits:", inits)
    print("max iterations: ", args.itermax)
    km_instance = kmedoids(dp, inits, data_type="distance_matrix", itermax=args.itermax)
    # print("running kmedoids")
    km_instance.process()
    # print("getting clusters")
    clusters = km_instance.get_clusters()
    predicts = [-1 for i in range(K)]
    for index, clust in enumerate(clusters):
        for pt in clust:
            predicts[pt] = index

    # print("cluster allocations: ", clusters)
    # print("predictions: ", predicts)
    # print("true labels: ", labels)
    score = adjusted_rand_score(labels, predicts)
    print("adj. rand score: ", score)
    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="remote_homology")
    ap.add_argument("--suffix", type=str, default="train")
    ap.add_argument("--cluster_label", type=str, default=None)
    ap.add_argument("--approach", type=str, default="euclid-pca")
    ap.add_argument("--itermax", type=int, default=1)
    ap = ap.parse_args()
    
    scores = []
    for _ in range(5):
        scores.append(main(ap))
    print("mean: {0:.4f}".format(sum(scores) / 5.0))


"""

Usage:
python pretrained_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 1
python pretrained_cluster_optim.py --cluster_label class_label --suffix train --itermax 1

"""