import argparse
import json
import pickle

import numpy as np
import math
from Bio.SubsMat.MatrixInfo import blosum62
from Bio import pairwise2
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

from sklearn.metrics import adjusted_rand_score
from numpy.random import default_rng

rng = default_rng()

# import seaborn as sns

# import matplotlib.pyplot as plt

fn_cnt = 0
dp = dict()

def get_distance(p1, p2):
    global fn_cnt
    global dp
    if fn_cnt % 1000 == 0:
        print("call count: ", fn_cnt)
    fn_cnt += 1
    if p1 in dp and p2 in dp[p1]: return dp[p1][p2]
    if p2 in dp and p1 in dp[p2]: return dp[p2][p1]
    if p1 not in dp:
        dp[p1] = dict()
    p1_p2 = pairwise2.align.globaldx(p1, p2, blosum62, score_only=True)
    p1_p1 = pairwise2.align.globaldx(p1, p1, blosum62, score_only=True)
    p2_p2 = pairwise2.align.globaldx(p2, p2, blosum62, score_only=True)
    dp[p1][p2] = p1_p1 + p2_p2 - 2.0 * p1_p2
    return dp[p1][p2]


def main(args):
    # np.random.seed(args.seed)
    global dp

    data_file = "../data/{0}/{0}_{1}.json".format(args.name, args.suffix)
    with open(data_file) as f:
        data = json.load(f)

    K = len(data)

    # K = 20
    print("full data len: ", K)
    proteins = [data[i]["primary"] for i in range(K)]

    s = dict()
    for i in range(len(proteins)):
        s[proteins[i]] = dict()

    for i in range(len(proteins)):
        s[proteins[i]][proteins[i]] = pairwise2.align.globaldx(proteins[i], proteins[i], blosum62, score_only=True)
        for j in range(i+1, len(proteins)):
            sim = pairwise2.align.globaldx(proteins[i], proteins[j], blosum62, score_only=True)
            s[proteins[i]][proteins[j]] = s[proteins[j]][proteins[i]] = sim

    for i in range(len(proteins)):
        dp[proteins[i]] = {}

    for i in range(len(proteins)):
        dp[proteins[i]][proteins[i]] = 0.0
        for j in range(i+1, len(proteins)):
            dist = s[proteins[i]][proteins[i]] + s[proteins[j]][proteins[j]] - 2.0 * s[proteins[i]][proteins[j]]
            dp[proteins[i]][proteins[j]] = dp[proteins[j]][proteins[i]] = dist

    labels = [int(data[i][args.cluster_label]) for i in range(K)]
    metric = distance_metric(type_metric.USER_DEFINED, func=get_distance)

    cluster_count = len(set(labels))
    print("num clusters: ", cluster_count)
    inits = rng.choice(K, size=cluster_count, replace=False)
    print("cluster inits:", inits)
    print("max iterations: ", args.itermax)
    km_instance = kmedoids(proteins, inits, metric=metric, itermax=args.itermax)
    print("running kmedoids")
    km_instance.process()
    print("getting clusters")
    clusters = km_instance.get_clusters()
    predicts = [-1 for i in range(K)]
    for index, clust in enumerate(clusters):
        for pt in clust:
            predicts[pt] = index

    print("cluster allocations: ", clusters)
    print("predictions: ", predicts)
    print("true labels: ", labels)
    print("adj. rand score: ", adjusted_rand_score(labels, predicts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="remote_homology")
    ap.add_argument("--suffix", type=str, default="train")
    ap.add_argument("--cluster_label", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--itermax", type=int, default=1)
    ap = ap.parse_args()
    main(ap)


"""

Usage:
python global_cluster.py --cluster_label class_label --suffix test_fold_holdout --itermax 1
python global_cluster.py --cluster_label class_label --suffix train --itermax 1

"""