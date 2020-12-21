import argparse
import json
import pickle
import os

import numpy as np
import math
from Bio.SubsMat.MatrixInfo import blosum62
from Bio import pairwise2
from pyclustering.cluster.kmedoids import kmedoids

from sklearn.metrics import adjusted_rand_score
from numpy.random import default_rng

rng = default_rng()


def main(args):
    data_file = "../data/{0}/{0}_{1}.json".format(args.name, args.suffix)
    with open(data_file) as f:
        data = json.load(f)

    K = len(data)
    
    print("full data len: ", K)
    proteins = [data[i]["primary"] for i in range(K)]

    if not os.path.exists("global_pairwise_dist_{0}_{1}.pkl".format(args.name, args.suffix)):

        print("calculating pairwise distances")
        s = [[0.0 for j in range(K)] for i in range(K)]
        for i in range(K):
            s[i][i] = pairwise2.align.globaldx(proteins[i], proteins[i], blosum62, score_only=True)
            for j in range(i+1, K):
                sim = pairwise2.align.globaldx(proteins[i], proteins[j], blosum62, score_only=True)
                s[i][j] = s[j][i] = sim
        print("similarity scores calculated")

        dp = [[0.0 for j in range(K)] for i in range(K)]
        for i in range(K):
            for j in range(i+1, K):
                dist = s[i][i] + s[j][j] - 2.0 * s[i][j]
                dp[i][j] = dp[j][i] = dist
        print("created dist matrix")
        
        with open("global_pairwise_dist_{0}_{1}.pkl".format(args.name, args.suffix), "wb") as handle:
            pickle.dump(dp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print("loading distances")
        with open("global_pairwise_dist_{0}_{1}.pkl".format(args.name, args.suffix), "rb") as handle:
            dp = pickle.load(handle)

    labels = [int(data[i][args.cluster_label]) for i in range(K)]
    
    cluster_count = len(set(labels))
    print("num clusters: ", cluster_count)
    inits = rng.choice(K, size=cluster_count, replace=False)
    print("cluster inits:", inits)
    print("max iterations: ", args.itermax)
    km_instance = kmedoids(dp, inits, data_type="distance_matrix", itermax=args.itermax)
    print("running kmedoids")
    km_instance.process()
    print("getting clusters")
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--itermax", type=int, default=1)
    ap = ap.parse_args()
    
    scores = []
    for _ in range(5):
        scores.append(main(ap))
    print("mean score: {0:.4f}".format(sum(scores) / 5.0))


"""

Usage:
python global_cluster_optim.py --cluster_label class_label --suffix test_fold_holdout --itermax 10
python global_cluster_optim.py --cluster_label fold_label --suffix test_fold_holdout --itermax 10

python global_cluster_optim.py --cluster_label class_label --suffix train --itermax 10
python global_cluster_optim.py --cluster_label fold_label --suffix train --itermax 10

"""